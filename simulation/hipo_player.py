from pypokerengine.players import BasePokerPlayer
import random as rand
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NB_SIMULATION = 200

class HipoPlayer(BasePokerPlayer):

    def __init__(self, model3J, mean3J, var3J, model2J, mean2J, var2J, coef_raise, coef):
        self.model3J = model3J
        self.mean3J = mean3J
        self.var3J = var3J
        self.model2J = model2J
        self.mean2J = mean2J
        self.var2J = var2J
        self.coef = torch.tensor(coef, dtype = torch.float64)
        self.coef_raise = coef_raise
        self.position = []
        self.loser = ""
        self.etat = {}
        self.etat["preflop"] = 0
        self.etat["flop"] = 0
        self.etat["turn"] = 0
        self.etat["river"] = 0
        self.etat["proba"] = 0
        self.etat["button"] = 0
        self.etat["small"] = 0
        self.etat["big"] = 0
        self.etat["value_bb"] = 0
        self.etat["my_stack"] = 0
        self.etat["stack_J2"] = 0
        self.etat["stack_J3"] = 0
        self.etat["my_bet"] = 0
        self.etat["bet_J2"] = 0
        self.etat["bet_J3"] = 0
        self.etat["pot"] = 0


    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        self.etat["proba"] = estimate_hole_card_win_rate(nb_simulation=NB_SIMULATION,nb_player=self.nb_player, hole_card=gen_cards(hole_card), community_card=gen_cards(community_card))
        
        pred = self.__pred_action()
        action, amount = self.__predtoaction(pred, valid_actions)      
        return action, amount

    def __pred_action(self):
        X = list(self.etat.values())
        if self.nb_player == 3:
            X = np.asarray(X)
            model = self.model3J
            X_scaled = []
            X_scaled.extend(X[:8])
            X_scaled.extend((X[8:] - self.mean3J) / np.sqrt(self.var3J))
        else:
            model, X_scaled = self.__3to2(X)
        
        X_tensor = torch.tensor([X_scaled], dtype=torch.float64)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).type(torch.cuda.DoubleTensor)
        X_tensor = X_tensor.to(device)
        outputs = model(X_tensor)
        outputs += self.coef.to(device)
        _, pred = torch.max(outputs.data, 1)
        return pred

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        if(self.nb_player != 3):
            print("ERROR : Le nombre de joueur n'est pas bon !")

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.loser = ""
        self.__get_seats(seats)
        

    def receive_street_start_message(self, street, round_state):
        self.__get_phase(street)
        self.__init_bet()
        self.etat["pot"] = round_state["pot"]["main"]["amount"]
        self.etat["value_bb"] = 20
        self.__get_position(round_state)
        self.__get_stack(round_state)
        

    def receive_game_update_message(self, new_action, round_state):
        self.__get_new_action(new_action)
        self.etat["pot"] = round_state["pot"]["main"]["amount"]
        # self.__get_seats(round_state["seats"])
        
            

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    
    def __get_phase(self, street):
        if(street == "preflop"):
            self.etat["preflop"] = 1
            self.etat["flop"] = 0
            self.etat["turn"] = 0
            self.etat["river"] = 0
        elif(street == "flop"):
            self.etat["preflop"] = 0
            self.etat["flop"] = 1
            self.etat["turn"] = 0
            self.etat["river"] = 0
        elif(street == "turn"):
            self.etat["preflop"] = 0
            self.etat["flop"] = 0
            self.etat["turn"] = 1
            self.etat["river"] = 0
        elif(street == "river"):
            self.etat["preflop"] = 0
            self.etat["flop"] = 0
            self.etat["turn"] = 0
            self.etat["river"] = 1

    def __init_bet(self):
        self.etat["my_bet"] = 0
        self.etat["bet_J2"] = 0
        self.etat["bet_J3"] = 0

    def __get_position(self, round_state):
        if(round_state["dealer_btn"] == 0):
            self.etat["button"] = 1
            self.etat["small"] = 0
            self.etat["big"] = 0
        if(round_state["small_blind_pos"] == 0):
            self.etat["button"] = 0
            self.etat["small"] = 1
            self.etat["big"] = 0
        if(round_state["big_blind_pos"] == 0):
            self.etat["button"] = 0
            self.etat["small"] = 0
            self.etat["big"] = 1
    
    def __get_stack(self, round_state):
        self.etat["my_stack"] = round_state["seats"][0]["stack"]
        self.etat["stack_J2"] = round_state["seats"][1]["stack"]
        if self.nb_player == 3 : self.etat["stack_J3"] = round_state["seats"][2]["stack"]

    def __get_new_action(self, new_action):
        if(new_action["action"] == ("raise" or "call")):
            index = 12 + self.position.index(new_action["player_uuid"])
            if(index == 12) : 
                bet = "my_bet"
            elif(index == 13) : 
                bet = "bet_J2"
            elif(index == 14) : 
                bet = "bet_J3"
            self.etat[bet] = new_action["amount"]
        # elif(new_action["action"] == "fold"):
        #     self.nb_player = 2
        
    def __get_seats(self, seats):
        self.nb_player = 3
        self.position.clear
        for seat in seats :
            if(seat["state"] == "participating"):
                self.position.append(seat["uuid"])
            elif(seat["state"] == "folded"):
                self.nb_player = 2
                self.loser = seat["uuid"]
            
    def __3to2(self, X):
        model = self.model2J
        del X[5] # Button feature
        index1 = 8 + self.position.index(self.loser)
        del X[index1] # stack du joueur éliminé
        index2 = 10 + self.position.index(self.loser)
        del X[index2] # bet du joueur éliminé
        X = np.asarray(X)
        X_scaled = []
        X_scaled.extend(X[:7])
        X_scaled.extend((X[7:] - self.mean2J) / np.sqrt(self.var2J))
        return model, X_scaled

    def __predtoaction(self, pred, valid_actions):
        if pred == 0 :
            choice = valid_actions[0]
            if valid_actions[1]["amount"] == 0 : choice = valid_actions[1]
        elif pred == 1:
            choice = valid_actions[1]
        elif pred == 2 : 
            choice = valid_actions[2]
        
        
        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            amount = min(amount["min"]*self.coef_raise, amount["max"])
        return action, amount

def setup_ai(): # Marche pas pour l'interface graphique
    ######### Réseau de neuronne #########
    class NN_3J(nn.Module):

        def __init__(self):
            super(NN_3J, self).__init__()
            self.largeur = 100
            self.batchnorm = nn.BatchNorm1d(self.largeur)
            self.input = nn.Linear(16, self.largeur)
            self.hidden = nn.Linear(self.largeur, self.largeur)
            self.final = nn.Linear(self.largeur, 4)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = self.final(x)
            return x

    model1 = NN_3J()
    model1.load_state_dict(torch.load('model_hyppo_3J.pth'))
    model1.eval()

    class NN_2J(nn.Module):

        def __init__(self):
            super(NN_2J, self).__init__()
            self.largeur = 100
            self.batchnorm = nn.BatchNorm1d(self.largeur)
            self.input = nn.Linear(13, self.largeur)
            self.hidden = nn.Linear(self.largeur, self.largeur)
            self.final = nn.Linear(self.largeur, 4)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = F.relu(self.batchnorm(self.hidden(x)))
            x = self.final(x)
            return x


    model2 = NN_2J()
    model2.load_state_dict(torch.load('model_hyppo_2J.pth'))
    model2.eval()
    #######################################
    return TiboPlayer(model3J = model1, model2J = model2)