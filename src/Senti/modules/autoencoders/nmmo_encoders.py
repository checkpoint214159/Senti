# maps from latent state to each action bucket in nmmo
# temporarily use nmmo_baseliens/agent_zoo/yaofeng

import pufferlib
import pufferlib.emulation
import pufferlib.models
import torch
import torch.nn.functional as F
from nmmo.entity.entity import EntityState
from torch import nn

from Senti.registry import ENCODERS

EntityId = EntityState.State.attr_name_to_col["id"]

def orthogonal_init(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, 0)


@ENCODERS.register_module()
class NmmoEncoders(nn.Module):  # TODO: change Policy from pufferlib for now?
    def __init__(self,
            intermediate=256, hidden_size=256, task_size=2048,):
        super().__init__()

        self.tile_encoder = TileEncoder(intermediate)
        self.player_encoder = PlayerEncoder(intermediate, hidden_size)
        self.item_encoder = ItemEncoder(intermediate, hidden_size)
        self.inventory_encoder = InventoryEncoder(intermediate, hidden_size)
        self.market_encoder = MarketEncoder(intermediate, hidden_size)
        # self.task_encoder = TaskEncoder(intermediate, hidden_size, task_size)
        # self.proj_fc = torch.nn.Linear(5 * intermediate, hidden_size)
        self.proj_fc = torch.nn.Linear(4 * intermediate, hidden_size)  # previously hardcoded to 5
        # because we take in the task encoded into intermediate. TODO make this more flex
        self.value_head = torch.nn.Linear(hidden_size, 1)
        orthogonal_init(self.proj_fc)
        orthogonal_init(self.value_head)

    def forward(self, env_outputs: dict):
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )

        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        inventory = self.inventory_encoder(item_embeddings)

        market_embeddings = self.item_encoder(env_outputs["Market"])
        market = self.market_encoder(market_embeddings)

        # task = self.task_encoder(env_outputs["Task"])

        # obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1)
        obs = torch.cat([tile, my_agent, inventory, market], dim=-1)
        obs = F.relu(self.proj_fc(obs))
        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_planes, img_size=(15, 15)):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
            torch.nn.LayerNorm((in_planes, *img_size)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
            torch.nn.LayerNorm((in_planes, *img_size)),
        )

    def forward(self, x):
        out = self.model(x)
        out += x
        return out


class TileEncoder(torch.nn.Module):
    def __init__(self, intermediate):
        super().__init__()
        self.type_embedding = torch.nn.Embedding(16, 62)  # hardcode to 16 types for now?
        # 62 concat with 2 gives us 64, which we pump into tile_resnet

        self.tile_resnet = ResnetBlock(64)
        self.tile_conv_1 = torch.nn.Conv2d(64, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, intermediate)
        self.tile_norm = torch.nn.LayerNorm(intermediate)
        orthogonal_init(self.tile_fc)

    def forward(self, tile):
        tile_position = tile[:, :, :2] / 128 - 0.5
        tile_type = tile[:, :, 2].long().clip(0, 15)
        tile = torch.cat((tile_position, self.type_embedding(tile_type)), dim=-1)
        agents, _, features = tile.shape
        # print('tile.shape, agents is first', tile.shape)
        tile = tile.transpose(1, 2).view(agents, features, 15, 15).float()
        
        latent = F.relu(self.tile_resnet(tile))
        # print('self.tile_resnet(tile)', latent.shape)
        latent = F.relu(self.tile_conv_1(latent))
        # print('conv1', latent.shape)
        latent = F.relu(self.tile_conv_2(latent))
        # print('conv2', latent.shape)
        latent = latent.contiguous().view(agents, -1)
        # print('configuous view reshape', latent.shape)
        latent = F.relu(self.tile_norm(self.tile_fc(latent)))
        # print('fc and norm', latent.shape)
        return latent


class MLPBlock(torch.nn.Module):
    def __init__(self, intermediate, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.model = [
            torch.nn.Linear(intermediate, hidden_size),
            torch.nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            self.model += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()]
        self.model.append(torch.nn.Linear(hidden_size, output_size))
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                orthogonal_init(layer)
        self.model = torch.nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out


class PlayerEncoder(torch.nn.Module):
    def __init__(self, intermediate, hidden_size):
        super().__init__()
        self.entity_dim = 31  # once again hardcoded for now
        self.player_offset = torch.tensor([i * 256 for i in range(self.entity_dim)])
        self.embedding = torch.nn.Embedding(self.entity_dim * 256, 32)

        self.EntityId = EntityState.State.attr_name_to_col["id"]
        self.EntityAttackerId = EntityState.State.attr_name_to_col["attacker_id"]
        self.EntityMessage = EntityState.State.attr_name_to_col["message"]
        self.id_embedding = torch.nn.Embedding(512, 64)
        self.embedding_idx = [self.EntityId, self.EntityAttackerId]
        self.no_embedding_idx = [i for i in range(self.entity_dim)]
        self.no_embedding_idx.remove(self.EntityId)
        self.no_embedding_idx.remove(self.EntityAttackerId)
        self.no_embedding_idx.remove(self.EntityMessage)

        self.agent_mlp = MLPBlock(64 + self.entity_dim - 3, hidden_size, hidden_size)
        self.agent_fc = torch.nn.Linear(hidden_size, hidden_size)
        self.my_agent_fc = torch.nn.Linear(hidden_size, intermediate)
        self.agent_norm = torch.nn.LayerNorm(hidden_size)
        self.my_agent_norm = torch.nn.LayerNorm(intermediate)
        orthogonal_init(self.agent_fc)
        orthogonal_init(self.my_agent_fc)

    def forward(self, agents, my_id):
        # Pull out rows corresponding to the agent
        agent_ids = agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        batch, agent, _ = agents.shape
        agent_embeddings = self.embedding(
            (agents[:, :, self.embedding_idx].long() + 256).clip(0, 511)
        ).reshape(batch, agent, -1)
        agent_embeddings = torch.cat(
            (agent_embeddings, agents[:, :, self.no_embedding_idx]), dim=-1
        ).float()
        agent_embeddings = F.relu(self.agent_mlp(agent_embeddings))

        my_agent_embeddings = agent_embeddings[torch.arange(agents.shape[0]), row_indices]
        agent_embeddings = F.relu(self.agent_norm(self.agent_fc(agent_embeddings)))
        my_agent_embeddings = F.relu(self.my_agent_norm(self.my_agent_fc(my_agent_embeddings)))
        return agent_embeddings, my_agent_embeddings


class ItemEncoder(torch.nn.Module):
    def __init__(self, intermediate, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 32)
        self.item_mlp = MLPBlock(2 * 32 + 12, hidden_size, hidden_size)
        self.item_norm = torch.nn.LayerNorm(hidden_size)

        self.discrete_idxs = [1, 14]
        self.discrete_offset = torch.Tensor([2, 0])
        self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        self.continuous_scale = torch.Tensor(
            [
                1 / 10,
                1 / 10,
                1 / 10,
                1 / 100,
                1 / 100,
                1 / 100,
                1 / 40,
                1 / 40,
                1 / 40,
                1 / 100,
                1 / 100,
                1 / 100,
            ]
        )

    def forward(self, items):
        if self.discrete_offset.device != items.device:
            self.discrete_offset = self.discrete_offset.to(items.device)
            self.continuous_scale = self.continuous_scale.to(items.device)

        # Embed each feature separately
        discrete = items[:, :, self.discrete_idxs] + self.discrete_offset
        discrete = self.embedding(discrete.long().clip(0, 255))
        batch, item, attrs, embed = discrete.shape
        discrete = discrete.view(batch, item, attrs * embed)

        continuous = items[:, :, self.continuous_idxs] / self.continuous_scale

        item_embeddings = torch.cat([discrete, continuous], dim=-1).float()
        item_embeddings = F.relu(self.item_norm(self.item_mlp(item_embeddings)))
        return item_embeddings


class InventoryEncoder(torch.nn.Module):
    def __init__(self, intermediate, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(12 * hidden_size, intermediate)
        self.norm = torch.nn.LayerNorm(intermediate)
        orthogonal_init(self.fc)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
        return F.relu(self.norm(self.fc(inventory)))


class MarketEncoder(torch.nn.Module):
    def __init__(self, intermediate, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, intermediate)
        self.norm = torch.nn.LayerNorm(intermediate)
        orthogonal_init(self.fc)

    def forward(self, market):
        return F.relu(self.norm(self.fc(market).mean(-2)))


class TaskEncoder(torch.nn.Module):
    def __init__(self, intermediate, hidden_size, task_size):
        super().__init__()
        self.fc = torch.nn.Linear(task_size, intermediate)
        self.norm = torch.nn.LayerNorm(intermediate)
        orthogonal_init(self.fc)

    def forward(self, task):
        return F.relu(self.norm(self.fc(task.clone().float())))
