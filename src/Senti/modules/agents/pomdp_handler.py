import random

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.func import vmap

from Senti.modules.agents.base import BaseAgentHandler, GenotypeStrategy
from Senti.modules.config.config import ConfigDict
from Senti.modules.dataclass.action_policy import ActionPolicy
from Senti.modules.dataclass.normal import Normal
from Senti.modules.dataclass.obs import LatentObservation
from Senti.modules.dataclass.pomdpstate import POMDPState
from Senti.modules.genome.preference_genome import PreferenceGenome
from Senti.modules.optim.dist_loss import DistributionLosses
from Senti.modules.optim.optimregistry import OptimRegistry
from Senti.modules.utils.caches import Cache, HierarchicalCache, create_POMDP_cache
from Senti.modules.utils.utils import nested_stack, param_traverse
from Senti.registry import AGENTS


@AGENTS.register_module()
class POMDPAgentHandler(BaseAgentHandler):

    _OPTIM_ATTRS = {
        'optimizer_registry':'_optimizer_registry',
        'loss_module': '_loss_module',
        'optim_config': '_optim_config'
    }  # dict sent to getattr, throws specific errors if
    # not inited that tells them these are runtime attrs

    def __init__(self,
        config: DictConfig,
    ):
        super().__init__(config)

        self.batch_size = self.config.batch_size
        self.history = self.config.history
        self.inference_steps = self.config.inference_steps
        self.atomic_timestep = self.config.atomic_timestep
        self.discrete_step = self.config.discrete_step
        self.planning_horizon = self.config.planning_horizon
        self.planning_steps = self.config.planning_steps
        self.n_policies_sampled = self.config.n_policies_sampled
        self.atomic_count = 0
        self.curr_timestep = 1
        self.param_learning_steps = self.config.param_learning_steps

        self.atomic_window = range(self.atomic_count - self.history, self.atomic_count)

        # caches in observation space
        observation_cache, action_cache, _cache = create_POMDP_cache()
        self.observation_cache: Cache = observation_cache
        self.action_cache: Cache = action_cache
        self._cache: HierarchicalCache = _cache
        self.prev_api: ActionPolicy | None = None
        
        self.state_dims = self.config.state_dims


    def load(self, strategy: dict | GenotypeStrategy, optim: bool = True):
        """
        load does the work to set up runtime related state. for instance you dont need optimizer and loss
        if you are running a custom method that doesnt use it. so we condition those under optim = True
        """
        super().load(strategy)

        self.curr_policy_dist: Normal = self.policy_init()

        if optim:
            self.optim_config: ConfigDict = self.config.optimizer     
            self.loss_module = DistributionLosses()
            self.optimizer_registry = OptimRegistry()
            
            inference_modules_substrings = [
                'obs_autoencoder', 'transition_model', 'zh_o', 'deliberative_head'
            ]
            self.optimizer_registry.register_optim(
                'inference',
                self.get_source_params(inference_modules_substrings),
                **self.optim_config.inference
            )

            wm_modules_substrings = ['grounding_wm', 'habitual_head']
            self.optimizer_registry.register_optim(
                'wm',
                self.get_source_params(wm_modules_substrings),
                **self.optim_config.wm
            )

            self.optimizer_registry.register_optim(
                'policy',
                list(self.curr_policy_dist.parameters()),
                **self.optim_config.policy
            )


    # @classmethod
    # def seed_preferences(self, pref_gene_dim: int, ids: list[int]):
    #     """
    #     Helper method to instantiate the preference genome
    #     This will be fed to the handler during the very first round of selection
    #     where our Selector is still empty.
    #     """
    #     return {
    #         id: PreferenceGenome(
    #             gene=nn.ParameterDict({'preferences': torch.nn.Parameter(torch.randn(pref_gene_dim))})
    #         ) for id in ids
    #     }


    def forward(self,
            observations: dict,
        ):

        t = self.curr_timestep
        self.observation_cache.add(t, observations)

        if (self.curr_timestep % self.atomic_timestep) == 0:
            self.atomic_count += 1
            timesteps = list(range(self.curr_timestep - self.atomic_timestep + 1, self.curr_timestep + 1))
            self.update_caches(timesteps, self.atomic_count)

            if (self.atomic_count % self.discrete_step) == 0 and self.atomic_count != 0:
                self.inference(
                    # self.inference_steps,
                    # self.param_learning_steps,
                    10, 2
                )
                fsergdty
                self.ground_wm()
                self.planning()
        
        self.curr_timestep += 1
        self.prune()















    # ------ cache related updating (before inference) ------
    def update_latent_obs(self, timesteps:list[int], atomic_t:int):
        """
        TODO: assumes that the obs encoder operates on each timestep seperately, i.e it doesnt support us just
        stacking the tensor and doing one pass through. this should change
        """
        obs_list = [self.observation_cache.get(t) for t in timesteps]
        obs_sequence = nested_stack(obs_list, dim=1)

        v_population = vmap(
            self.agent_blueprint.f_update_latent_obs,
            in_dims=(0, 0)
        )

        # detaches to prevent double grad problem
        population_atomic = v_population(self.merged_params, obs_sequence).detach()

        print('population_atomic shape?', population_atomic.shape)
        self._cache.set_container(
            'latent_obs', 
            atomic_t, 
            LatentObservation(lat_o=population_atomic.to(self.device))
        )

    def update_states_cache(self, atomic_t: int, obs: torch.Tensor):
        prev_state: POMDPState | None = self._cache.get_semantic('s', atomic_t - 1) if self._cache.has_semantic('s', atomic_t - 1) else None
        prev_action: torch.Tensor | None = self._cache.get_semantic('a', atomic_t - 1) if self._cache.has_semantic('a', atomic_t - 1) else None
        in_dims = [0, 0, 0, None, None]  # params, obs, prev_state, prev_action, batch_size, atomic_t

        # hardcode to change to None if prev_state / prev_action is None
        in_dims[2] = None if prev_state is None else 0
        in_dims[3] = None if prev_action is None else 0
        
        v_population = vmap(
            self.agent_blueprint.f_pred_new_state,
            in_dims=tuple(in_dims),
            randomness='same',
        )
        h, z = v_population(
            self.merged_params, obs, prev_state, prev_action, atomic_t
        )
        h = h.clone(detach=True) # VERY CRUCIAL TO PREVENT DOUBLE GRAD PROBLEM
        z = z.clone(detach=True) # VERY CRUCIAL TO PREVENT DOUBLE GRAD PROBLEM

        self._cache.set_container('states', atomic_t, 
            POMDPState(h=h, z=z, as_parameter=True).to(self.device)
        )

    def update_caches(self, timesteps:list[int], atomic_t:int):
        """
        calls various cache updating methods.
        """
        self.update_latent_obs(timesteps, atomic_t)
        if not self._cache.has_container('api', atomic_t):
            self.update_api_cache(atomic_t)  # TODO: decide to set here or immediately after planning.
            # here is neater, but technically if we freeze the agent after we make its move, it will have had
            # no idea what its last latent aciton was.
        if not self._cache.has_container('states', atomic_t):
            obs: torch.Tensor = self._cache.get_semantic('lat_o', atomic_t)
            self.update_states_cache(atomic_t, obs)

    def update_api_cache(self, atomic_t: int):
        if self.prev_api is None:
            a, pi = self.empty_action_init(), self.curr_policy_dist.sample()
            prev_api = ActionPolicy(
                a=a,
                pi=pi,
            )
        else:
            prev_api = self.prev_api
        self._cache.set_container('api', atomic_t, prev_api)
        














    # --------- inference related functions ----------
    def belief_optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer for over beliefs over states
        all_params = self._cache.get_params_flattened('states')
        self.optimizer_registry.register_optim('states_cache', all_params, optim_name='SGD', **self.optim_config.belief)


    def inference(
        self,
        max_update_steps: int = 10,
        param_learning_steps: int = 2,
    ) -> float:
        """
        Run the inference procedure for belief updates and model parameter learning.
        Currently in the form of amortized inference, perhaps when I have time will fit in non-amortized (param learning
        only when VFE converges.)

        Args:
            max_update_steps (int): Maximum number of message passing steps to perform.
            param_learning_steps (int): Number of belief updates before checking VFE and (possibly) updating model parameters.
        Returns:
            float: Final VFE after inference and (optional) learning.
        """

        for step in range(max_update_steps):
            if step == 0:  # first step, percieve for latest prior and wrap in optim
                self.belief_optim_wrap()
                
            # random timestep group selection, to run inference on
            timestep = random.choice(list(self._cache.keys('states')))
            print('-----------Selected atomic timestep---------------', timestep)
            self.inference_step(timestep,
                # P_states,
                backprop_belief=True)  # lr scheduling comes later. test first
            self._cache.map_cache('latent_obs', lambda c: c.clone(detach=True))
            # self._cache.map_cache('api', lambda c: c.clone(detach=True))

            # every 'param_learning_steps', do param learning (backprop_belief=False)
            if step % param_learning_steps == 0 and step != 0:
                before_dict = param_traverse('z_given_h.0.weight', self.merged_params)
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        # P_states=P_states,
                        backprop_belief=False
                    ) for t in self._cache.keys('states')
                )
                self._cache.map_cache('latent_obs', lambda c: c.clone(detach=True))

                self.optimizer_registry.do_step(
                    key='inference',
                    loss=total_vfe
                )
                after_dict = param_traverse('z_given_h.0.weight', self.merged_params)
                diff_dict = {
                    k: a - before_dict[k] for k, a in after_dict.items()
                }
                print('diff_dict', diff_dict)

        return total_vfe
    
    def inference_step(
            self,
            timestep,
            backprop_belief=False
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        Args:
            timestep: Selected timestep with which we perform inference for
            P_states: Dict of Ph_t 
            backprop_belief: Boolean on whether or not to call belief optimizer.
        """
        t = timestep
        s_t: POMDPState = self._cache.get_semantic('s', t)

        # p(z_t | h_t) kl q(z_t | h_t, o_t)
        qh_t = s_t.get('h')
        qz_t = s_t.get('z')

        print('gradients enabled before vmap:', [p.requires_grad for p in self.merged_params.values()])
        pz_t = vmap(
            self.agent_blueprint.f_wm_predict_z,
            in_dims=(0,0),
            randomness='same',
        )(self.merged_params, qh_t, wm='transition_model')
        if not backprop_belief:
            print('Layer weight of z_given_h?', param_traverse('z_given_h', self.merged_params))
        print('PZ_T RAGHHH', pz_t.shape)
        print('pz_t has grad_fn??', pz_t.mean.grad_fn)
        self.loss_module.include(
            ('qz_t vs pz_t', self.loss_module.kl,
            qz_t, pz_t))
        testpz_t = self.loss_module.compute()
        print('INITIAL PZ_T OUTCOME HAS GRADFN??', testpz_t.grad_fn)
        # sgdhgd
        # if testpz_t.grad_fn is None:
        #     raise NotImplementedError("KABOOM")
        self.loss_module.include(
            ('qz_t vs pz_t', self.loss_module.kl,
            qz_t, pz_t))

        tm1 = t - 1
        if self._cache.has_semantic('s', tm1) and self._cache.has_semantic('a', tm1):
            s_tm1 = self._cache.get_semantic('s', tm1)
            a_tm1 = self._cache.get_semantic('a', tm1)
            # -------- how well does posterior from t-1 predict posterior of t -------- 
            _, _, pz_t_from_tm1 = vmap(
                self.agent_blueprint.f_wm_full_forward,
                in_dims=(0,0,0),
                randomness='same',
            )(self.merged_params, s_tm1, a_tm1, wm='transition_model')
            print('pz_t_from_tm1 RAGHHH', pz_t_from_tm1.shape)
            # self.transition_model.forward(s_tm1, a_tm1)
            self.loss_module.include(
                ('qz_t vs tm1 -> pz_t', self.loss_module.kl,
                qz_t, pz_t_from_tm1))
            test_check_grad = self.loss_module.compute()
            print('tm1 interaction?', test_check_grad)
            print('has grad from tm1 interaction?', test_check_grad.grad_fn)
            
            # ------- loss from action or something lol --------
            # TODO do we try and do this? will have to predict a from pz_t_from_tm1, and ph_t_from_tm1, which may be mroe unstable?
        
        tp1 = t + 1
        if self._cache.has_semantic('s', tp1) and self._cache.has_semantic('a', t):
            qz_tp1 = self._cache.get_semantic('z', tp1)
            a_t = self._cache.get_semantic('a', t)
            # _, _, pz_at_tp1= self.transition_model.forward(s_t, a_tp1)
            _, _, pz_at_tp1 = vmap(
                self.agent_blueprint.f_wm_full_forward,
                in_dims=(0,0,0),
                randomness='same',
            )(self.merged_params, s_t, a_t,  wm='transition_model')
            self.loss_module.include(
                ('qz_tp1 vs t -> pz_tp1', self.loss_module.kl,
                qz_tp1, pz_at_tp1))

        states_energy = self.loss_module.compute()
        # print('states energy grad fn?', states_energy.grad_fn)
        # if states_energy.grad_fn is None:
        #     raise NotImplementedError("EXPLODEA!!!!")

        # message from obs
        lat_o_pred = vmap(
            self.agent_blueprint.zh_o_forward,
            in_dims=(0,0,0),
            randomness="same",
        )(self.merged_params, qz_t, qh_t)
        lat_o = self._cache.get_semantic('lat_o', t)

        obs_MSE = self.loss_module.MSE(lat_o, lat_o_pred)  # positive equivalent for NLL if latent obs
        # was a variational belief. but it isnt, its just some tensors
        # it should help to update zh_o and qz_t and qh_t though
        
        total_energy = states_energy + obs_MSE

        if backprop_belief:
            self.optimizer_registry.do_step(
                key='states_cache',
                loss=total_energy
            )
        
        return total_energy
    











    def ground_wm(self):
        """
        Updates predictive WM to bootstrap it to transition model.
        Since during planning we operate entirely within imagination space, no observation semantics
        can be spotted here
        """
        # format x
        print('-----------------GROUNDING WM STEP-----------------')
        all_a = self._cache.get('api') \
            .vmap(lambda s: s.get('a')) \
            .values()
        all_z = self._cache.get('states') \
            .vmap(lambda s: s.get('z')) \
            .values()
        all_h = self._cache.get('states') \
            .vmap(lambda s: s.get('h')) \
            .values()
        
        all_z = type(all_z[0]).concat(all_z, 1)
        z = all_z.map(lambda z: z[:, :-1])  # exclude the final state when passing to wm

        # TODO This is horrible practice, hardcoding [:-1] like that 
        a = torch.concat(all_a[:-1], 1)
        seed_h = all_h[0]
        s = POMDPState(
            h=seed_h,
            z=z,  # [B, L-1, z_emb]
            as_parameter=False,  # TODO verify this again? does this annihilate gradients or smth
        )

        # forward
        _, pred_h, pred_z = vmap(
            self.agent_blueprint.f_wm_full_forward,
            in_dims=(0,0,0),
            randomness='same',
        )(self.merged_params, s, a,  wm='grounding_wm')

        # correct wm to accurately predict our beliefs, based only off h.
        ground_z = all_z.map(lambda z: z[:, 1:])  # get all except first state
        self.loss_module.include(
            ('wm_grounding', self.loss_module.kl, 
            pred_z, ground_z),
        )

        pred_a = vmap(
            self.agent_blueprint.habitual_head_forward,
            in_dims=(0,0,0),
            randomness='same',
        )(self.merged_params, pred_z, pred_h)
        print('pred_a gradfn in ground_wm?', pred_a.grad_fn)
        action_mse = self.loss_module.MSE(pred_a, a)

        grounding_loss = self.loss_module.compute() + action_mse

        self.optimizer_registry.do_step(
            key='wm',
            loss=grounding_loss)



    def planning(
        self,
    ):
        """
        do planning: sample n policies, run rollout function with it.
        Repeat this until EFE converges or we run out of steps

        Then, use this updated policy to predict action
        Use an affine transformation to augment the pre-sampling policy

        Start at t-1, because we dont have action for the current step, so
        we must take the previous action
        """
        # first, create planning priors from wm
        seed_s = self._cache.get_semantic('s', self.atomic_count - 1)
        seed_h, seed_z = seed_s.get('h'), seed_s.get('z')
        seed_a = vmap(
            self.agent_blueprint.habitual_head_forward,
            in_dims=(0,0,0),
            randomness='same',
        )(self.merged_params, seed_z, seed_h)
        print('seed_a gradfn in ground_wm?', seed_a.grad_fn)

        s, a = seed_s, seed_a
        priors = [s]
        # freeze gradients for prior_z
        print("NO WITHOUT GRAD CONTEXT HERE!!")
        for _ in range(self.planning_horizon):  # predict t+1, t+2, ...
            _, h, z = vmap(
                self.agent_blueprint.f_wm_full_forward,
                in_dims=(0,0,0),
                randomness='same',
            )(self.merged_params, s, a,  wm='grounding_wm')
            
            a = vmap(
                self.agent_blueprint.habitual_head_forward,
                in_dims=(0,0,0),
                randomness='same',
            )(self.merged_params, z, h)
            print('a gradfn in planning?', a.grad_fn)

            s = POMDPState(
                h=h,
                z=z,
                as_parameter=False,
            )
            priors.append(s)
        prior_z = [s.get('z') for s in priors]

        # old_policy = self.curr_policy_dist.clone(detach=True)

        for i in range(self.planning_steps):
            self.planning_step(seed_s, seed_a, prior_z)

        # lastly, sample updated policy
        final_pi = self.curr_policy_dist.sample()
        final_a = vmap(
            self.agent_blueprint.deliberative_head_forward,
            in_dims=(0,0,0,0),
            randomness='same',
        )(self.merged_params, seed_z, seed_h, final_pi)

        self.prev_api = ActionPolicy(
            a=final_a.detach(),
            pi=final_pi.detach()
        )    

        # TODO merge new policy and old one tgt


    def planning_step(
        self,
        seed_s,
        seed_a,
        prior_z,
    ):  
        efe = 0
        for _ in range(self.n_policies_sampled):
            pi = self.curr_policy_dist.sample()
            print('pi?', pi)
            rollout = self.rollout(seed_s=seed_s, seed_a=seed_a, policy=pi)
            print('pi after rollouts??', pi)
            rollout_z = [s.get('z') for s in rollout]
            print('rollout_z grads?', [z.mean.grad_fn for z in rollout_z])
            print('prior_z grads?', [z.mean.grad_fn for z in prior_z])
            [self.loss_module.include(
                ('instrumental_value', self.loss_module.kl, 
                rz, pz),
            ) for rz, pz in zip(rollout_z, prior_z)]
            instrumental = self.loss_module.compute()

            epistemic = sum([s.get('z').entropy() for s in rollout])
            print('instrumental?', instrumental.grad_fn, 'epistemic', epistemic.grad_fn)
            efe += instrumental - epistemic
            print('efe grad fn?', efe.grad_fn)
            
        self.optimizer_registry.do_step(
            key='policy',
            loss=efe
        )

    def rollout(
            self,
            seed_s,
            seed_a,
            policy,
        ):
        """
        Using the grounding wm as our prior, sample various policies and derive EFE for each one from rollouts
        """
        # create planning priors from wm.
        rollout = [seed_s]
        s, a = seed_s, seed_a
        # print('seed_s', seed_s)
        for _ in range(self.planning_horizon):
            _, h, z = vmap(
                self.agent_blueprint.f_wm_full_forward,
                in_dims=(0,0,0),
                randomness='same',
            )(self.merged_params, s, a, wm='transition_model')
            print('gradient fn of z in rollout?', z.mean.grad_fn)

            a = vmap(
                self.agent_blueprint.deliberative_head_forward,
                in_dims=(0,0,0,0),
                randomness='same',
            )(self.merged_params, z, h, policy)

            s = POMDPState(
                h=h,
                z=z,
                as_parameter=False,
            )
            print('grad of a?', a.grad_fn)
            rollout.append(s)

            print('-----------TESTING SAMPLING---------------')
            print('z mean has grad after sampling?')
            testz = z.sample()
            print('z_mean grad fn?', z.mean.grad_fn)
            print('policy dist has grad after sampling?')
            testpolicygrad = self.curr_policy_dist.sample()
            print('curr_policy_dist mean grad fn?', self.curr_policy_dist.mean.grad_fn)

        return rollout










    # ------- utilities ---------
    def empty_action_init(self) -> torch.Tensor:
        """helper method to sloppily create empty a."""
        return torch.zeros(self.batch_size, 1, self.state_dims.a_dim,).to(self.device) # TODO fix the magic number. it really is supposed to be 1,
        # to represent the singular atomic timestep, but this is horrible practice.


    def policy_init(self) -> Normal:
        v_population = vmap(
            self.agent_blueprint.f_predict_policy,
            in_dims=(0,),
            randomness='same',
        )

        policy: Normal = v_population(self.merged_params).parameterize().clone(detach=True)
        return policy
    
    def prune(self):
        """
        Helper func to prune from caches if not in atomic_window.
        """
        new_atomic_window = range(self.atomic_count - self.history, self.atomic_count)
        removed_atomic_timesteps = list(set(self.atomic_window) - set(new_atomic_window))
        self.atomic_window = new_atomic_window
        interleaved = [
            sub_step 
            for x in removed_atomic_timesteps
            for sub_step in range((x - 1) * self.atomic_timestep + 1, x * self.atomic_timestep + 1)
        ]

        # from removed timesteps, just assume that for each element, subtract atomic_timesteps number
        # to get the actual timesteps to remove from obs cache.

        print('REMOVED ATOMIC TIMESTEPS', removed_atomic_timesteps)

        for t in removed_atomic_timesteps:
            self._cache.remove_container('states', t) \
                if self._cache.has_container('states', t) else None
            self._cache.remove_container('api', t) \
                if self._cache.has_container('api', t) else None
            self._cache.remove_container('latent_obs', t) \
                if self._cache.has_container('latent_obs', t) else None

        for t in interleaved:
            self.observation_cache.remove(t) if self.observation_cache.has(t) else None

