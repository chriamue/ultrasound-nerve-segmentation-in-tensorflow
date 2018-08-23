import os
import sys
import yaml
from train import nerve_train


def main(run='run', results_dir="results/", batch_size=1, epochs=5, config={}, mode='--train'):
    nerve_train.train(run=run, results_dir=results_dir, batch_size=batch_size, epochs=epochs, config=config)
    

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        configs = yaml.load(file)
        for config in configs:
            c = configs[config]
            results_dir = "results/"+config+"/"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            with open(results_dir + 'config.yml', 'w') as outfile:
                yaml.dump({config: c}, outfile, default_flow_style=False)
            main(run=config, results_dir=results_dir, batch_size=c['batch_size'],
                 epochs=c['epochs'], config=c)