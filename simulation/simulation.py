from pypokerengine.api.game import setup_config, start_poker
from callbot import CallBot
from raisebot import RaiseBot
from consoleplayer import ConsolePlayer
from randombot import RandomBot 
from honestplayer import HonestPlayer
from hipo_player import HipoPlayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

######### Réseau de neuronne #########
class NN_3J(nn.Module):

    def __init__(self):
        super(NN_3J, self).__init__()
        self.largeur = 100
                
        self.input = nn.Linear(16, self.largeur)
        self.hidden = nn.Linear(self.largeur, self.largeur)
        self.final = nn.Linear(self.largeur, 3)

    def forward(self, x):

        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.final(x)
        return x


class NN_2J(nn.Module):

    def __init__(self):
        super(NN_2J, self).__init__()
        self.largeur = 100
        self.input = nn.Linear(13, self.largeur)
        self.hidden = nn.Linear(self.largeur, self.largeur)
        self.final = nn.Linear(self.largeur, 3)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = self.final(x)
        return x

#######################################



def main(coef_raise, coef):
    model1 = NN_3J()
    model1.load_state_dict(torch.load('model_hyppo_3J.pth'))
    model1.eval()
    mean3J = np.asarray([25.62989439, 506.31789281, 500.85171349, 492.83039371, 8.46048567,  29.11114304,  35.52415763, 127.08888929])
    var3J = np.asarray([77.21836346, 20549.21166419, 23143.48462078, 23994.43942467, 208.35494771,  6671.09516214,  8899.36275071, 22232.29371878])

    model2 = NN_2J()
    model2.load_state_dict(torch.load('model_hyppo_2J.pth'))
    model2.eval()
    mean2J = np.asarray([37.99850064, 665.80849344, 692.35777389, 15.20394244, 56.24048568, 160.47079922])
    var2J = np.asarray([226.25168453, 59823.15358233, 91017.39574507, 440.05912923, 23585.63323956, 41242.62439903])

    config = setup_config(max_round=1000, initial_stack=500, small_blind_amount=10)
    config.register_player(name="h1", algorithm=HipoPlayer(
        model3J = model1, mean3J = mean3J, var3J = var3J, 
        model2J = model2,  mean2J = mean2J, var2J = var2J,
        coef_raise = coef_raise, coef = coef))
    #config.register_player(name="r1", algorithm=RandomPlayer())
    config.register_player(name="f1", algorithm=CallBot())
    config.register_player(name="f2", algorithm=CallBot())

    game_result = start_poker(config, verbose = 0)

    if(game_result["players"][0]["stack"] > 1000):
        return 1
    else:
        return 0
    

# A FAIRE : 
# - Faire fonctionner avec l'interface graphique

nb_simulations = 1000
coefs = [[-4, 3, -2], ]
coefs_raise = [4]


for coef_raise in coefs_raise:
    for coef in coefs :
        wins = 0
        for i in range(nb_simulations):
            # print(i)
            wins += main(coef_raise, coef)
            # os.system('cls')
        print("*************")
        print(coef_raise, ' | ', coef, ': ', wins/nb_simulations*100,"%")
        print("*************")