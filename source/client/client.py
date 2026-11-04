import subprocess

from tokenizer.tokenizer import Tokenizer
from dataset.dataset import TextDataset
from training.train import Train
from training.generate import Generate


"""
Shell of the hub for bert-gpt
"""
class Client:

    def __init__(self):
        self.tag = f'''
██████╗ ███████╗██████╗ ████████╗    ██████╗ ██████╗ ████████╗
██╔══██╗██╔════╝██╔══██╗╚══██╔══╝   ██╔════╝ ██╔══██╗╚══██╔══╝
██████╔╝█████╗  ██████╔╝   ██║█████╗██║  ███╗██████╔╝   ██║   
██╔══██╗██╔══╝  ██╔══██╗   ██║╚════╝██║   ██║██╔═══╝    ██║   
██████╔╝███████╗██║  ██║   ██║      ╚██████╔╝██║        ██║   
╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝        ╚═╝   
** custom GPT developed only with numpy and python **

'''     
        self.options = {
            1: "Prepare Data",
            2: "Train Data", 
            3: "Load Graphs",
            4: "Talk"
        }


    def fetch_checkpoint(self) -> str: # move this into the train.py more than likely
        data_train_date = ""
        data_name = ""
        return f""
    
    def opt_one(self):
        pass

    def opt_two(self):
        pass

    def opt_three(self):
        pass

    def opt_four(self):
        pass

    def main(self):
        print(self.tag)
        for i in self.options:
            print(f"[{i}] {self.options[i]}")

        inp = input("\n<: ")

        match inp:
            case 1:
                self.opt_one()
            case 2:
                self.opt_two()
            case 3:
                self.opt_three()
            case 4:
                self.opt_four()


if __name__ == "__main__":
    client = Client()

    client.main()

