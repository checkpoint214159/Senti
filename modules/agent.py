class Agent(nn.Module):
    """
    An agent built on the concepts of active inference.

    For now, only amortized inference will be supported. For research purposes i will try and make non-amortized easily
    integrable with the overall flow.
    """

    def __init__(self,
        clip_config=None,
        num_policies=None,
    ):
        # admin stuff
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path = clip_config.pop('ckpt_path', None)
        self.amortized_inf = True

        # observations and states cache.
        self.states_cache = {}

        self.observation_cache = {}
        self.latent_observation_cache = {}

        # encoder to encode incoming observation(s)
        # self.encoder = MineCLIP(**clip_config).to(device)
        # self.encoder.load_ckpt(path, strict=True)
        # logging.info("Successfully loaded MineCLIP encoder.")
        self.encoder = nn.Linear(2, 2)

        self.state_encoder = nn.Linear(2, 2)  # P(s|o)

        # maps latent observation to latent states
        self.decoder = nn.Linear(2, 2) # P(o|s).

        self.preference_net = 1
        self.interpreter = 1
        self.transition_model = nn.Linear(2, 2)

        params = (
            list(self.state_encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.transition_model.parameters())
        )
        self.model_optimizer = torch.optim.SGD(params, lr=0.05)
                
        self.num_policies = num_policies
        self.state_dicts = {}

        self.curr_timestep = 0

    @staticmethod
    def kl_divergence(mean1, mean2):
        """
        Helper func to calculate kl_divergence between two isotropic gaussians with variance 1, for now
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2)  # what even. idk bro, if gaussian is isotropic multivariate, its just this?
    
    @staticmethod
    def log_likelihood(o_true, o_pred):
        """
        Helper func to calculate log_likeihood of pred, w.r.t truth. Equivalent to above method for isotropic gaussians with
        var 1 too, see how we can change this
        """
        # assume we have uniform multivariate gaussian
        dist = torch.distributions.Normal(o_true, 1.0)
        return dist.log_prob(o_pred)
    
    @staticmethod
    def param_learning(optimizer, loss):
        """
        Simple helper method to update arbitrary optimizer (just so i dont need to repeat code)
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def encode_obs(self):
        """
        Simple helper method to update all timesteps in observation cache.
        Computationally expensive, TODO efficienitize this
        """
        for t, o in self.observation_cache.items():
            lat_o = self.encoder(o)  # revert this back to MineCLIP encoding form
            self.latent_observation_cache[t] = lat_o

    def init_var_beliefs(self):
        """
        Simple helper method to initialize all variational beliefs over states, qs_t.
        """
        for t, lat_o in self.latent_observation_cache.items():
            self.states_cache[t] = self.state_encoder(lat_o)

    def optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        for t in self.states_cache:
            self.states_cache[t] = torch.nn.Parameter(self.states_cache[t])

        # seperate optimizer over beliefs
        all_params = list(self.states_cache.values())
        self.beliefs_optimizer = torch.optim.SGD(all_params, lr=0.5)

    def forward(self,
                observations,
            ):
        """
        Order of execution:
            Perception:
                1. Encode current observation into a latent observation.
                2. Encode latent observation into belief about latent state at current timestep t.
            Belief updating:
                3. From existing beliefs in the belief cache, calculate VFE and do belief updates, to minimize
                    VFE
            Parameter learning:
                4. Final VFE acts as model loss to update parameters
            Planning:
                5. Sample the several policies from the new model.
                6. Within each policy, calculate EFE.
                7. Update distribution over policies
                8. Repeat 5-7 until convergence.
            Action:
                8. Sample a policy from the final distribution over policies, then take action
        Then increment timestep by 1.
        """
        # 1 - 4:
        t = self.curr_timestep
        self.observation_cache[t] = observations  # cache this observation
        self.inference()
        # update parameters via vfe sum as loss.

        # 
        # converged = False
        # while not converged:
        #     for i in range(self.num_policies):
        #         pi = Q.sample  # some sample method
                
        #         # minimize vfe

        # # 8 - 10:
        # policy = Q.sample
        # action = policy.sample()

        # self.curr_timestep += 1

        # return action

        self.curr_timestep += 1

    def planning(
            self,
            policy,
        ):
        raise NotImplementedError

    def inference(
        self,
        max_update_steps: int = 10000,
        update_rounds: int = 10,

    ) -> float:
        """
        Run the inference procedure for belief updates and model parameter learning.
        Currently in the form of amortized inference, perhaps when I have time will fit in non-amortized (param learning
        only when VFE converges.)

        Args:
            max_update_steps (int): Maximum number of message passing steps to perform.
            update_rounds (int): Number of belief updates before checking VFE and (possibly) updating model parameters.
        Returns:
            float: Final VFE after inference and (optional) learning.
        """
        # save prior beliefs as P (fixed reference)
        # prev_vfe = float('inf')
        # stable_steps = 0
        amortized_reset = True

        for step in range(max_update_steps):
            # for every amortized reset called, which is after one parameter learning session, recompute latent obs and 
            # beliefs over states, then optim wrap them.
            if amortized_reset:
                self.encode_obs()
                self.init_var_beliefs()
                self.optim_wrap()
                amortized_reset = False
            if step == 0:  # first step, initialize frozen priors.
                P_states = {k: s.clone() for k, s in self.states_cache.items()}

            # random timestep selection, to run inference on
            timestep = random.choice(list(self.states_cache))
            print('Selected timestep', timestep)
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = {k: s.clone().detach() for k, s in P_states.items()}
            self.latent_observation_cache = {k: lat_o.clone().detach() for k, lat_o in self.latent_observation_cache.items()}

            # every 'update_rounds', do param learning
            if step % update_rounds == 0:
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        P_states=P_states,
                        backprop_belief=False
                    ) for t in self.states_cache
                )
                P_states = {k: s.clone().detach() for k, s in P_states.items()}  # detaching is very important or pytorch explodes
                print(f"[Step {step}] VFE = {total_vfe:.6f}")

                if self.amortized_inf:
                    # param learning for amortized case
                    self.param_learning(
                        optimizer=self.model_optimizer,
                        loss=total_vfe
                    )

                # else:
                #     # check for VFE convergence in non-amortized case
                #     if abs(prev_vfe - current_vfe) < vfe_tol:
                #         stable_steps += 1
                #         if stable_steps >= patience:
                #             print("Converged.")
                #             self.param_learning(current_vfe)
                #             return current_vfe
                #     else:
                #         stable_steps = 0  # Reset if VFE jumped

                #     prev_vfe = current_vfe

        print("Reached max inference steps.")
        return total_vfe
    
    def compute_vfe_node(
        self,
        ps_t: torch.Tensor,
        qs_t: torch.Tensor,
        po_t: torch.Tensor,
        o_pred: torch.Tensor,
        prev_ps_t: torch.Tensor | None = None,
        next_qs_t: torch.Tensor | None = None,
        variance_prior: float = 1.0,
        variance_obs: float = 1.0
    ) -> torch.Tensor:
        """
        Computes the variational free energy (VFE) at a specific timestep `t`.

        Args:
            ps_t (torch.Tensor): Prior state at timestep t (frozen since perception)
            qs_t (torch.Tensor): Variational est. state at timestep t
            po_t (torch.Tensor): Observation at timestep t.
            o_pred (torch.Tensor): Predicted observation at timestep t (decoder output)
            prev_ps_t (torch.Tensor, optional): Prior at previous timestep t - 1 (used to calculate posterior at t)
            next_qs_t (torch.Tensor, optional): Variational posterior at timestep t-1
            variance_prior (float): Variance of the prior (assumes isotropic Gaussian).
            variance_obs (float): Variance of the observation model (isotropic Gaussian).

        Returns:
            torch.Tensor: Scalar VFE value at timestep t.
        """
        # Compute gradient of prediction error w.r.t. state  (right now simple form TODO not simple form?)
        energy_obs = self.log_likelihood(po_t, o_pred)
        # print('energy obs', energy_obs)

        # transition messages
        energy_states = 0.0
        
        # msg from previous timestep
        if prev_ps_t is not None:
            # print('prev msg', self.kl_divergence(qs_t, prev_ps_t))
            energy_states += self.kl_divergence(qs_t, prev_ps_t)

        # msg from kl at next timestep
        if next_qs_t is not None:
            ps_t_next = self.transition_model(ps_t)
            # print('next msg', self.kl_divergence(next_qs_t, ps_t_next))
            energy_states += self.kl_divergence(next_qs_t, ps_t_next)

        # msg from prior
        # print('curr prior msg', self.kl_divergence(qs_t, ps_t))
        energy_states += self.kl_divergence(qs_t, ps_t)

        total_energy = (energy_states - energy_obs).sum()

        return total_energy
    
    def inference_step(
            self,
            timestep,
            P_states,
            backprop_belief=False
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        For each step of belief updating, we isolate one node, calculate the corresponding messages
        fed to it, then update it. Messages include
            - observation message
            - prev state message
            - future state message

        Args:
            timestep: Selected timestep with which we perform inference for
            P_states: Dict of Ps_t 
            backprop_belief: Boolean on whether or not to call belief optimizer.
        """
        qs_t = self.states_cache[timestep]
        ps_t = P_states[timestep]
        
        # observation messages
        o_pred = self.decoder(qs_t)
        o_true = self.latent_observation_cache[timestep]

        # posterior qs_t-1
        prev_ps_t = None
        if (timestep - 1) in self.states_cache:
            s_prev = self.states_cache[timestep - 1]
            prev_ps_t = self.transition_model(s_prev)

        # prior at qs_t+1
        next_qs_t = None
        if (timestep + 1) in self.states_cache:
            next_qs_t = self.states_cache[timestep + 1]

        total_energy = self.compute_vfe_node(
            ps_t=ps_t, 
            qs_t=qs_t, 
            po_t=o_true,
            o_pred=o_pred,
            prev_ps_t=prev_ps_t,
            next_qs_t=next_qs_t,
        )

        if backprop_belief:
            self.param_learning(optimizer=self.beliefs_optimizer, loss=total_energy)

        return total_energy
