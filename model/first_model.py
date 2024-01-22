from datasets import load_dataset
import pandas as pd

imagenette = load_dataset('frgfm/imagenette', 'full_size', split='train[:10%]', trust_remote_code=True)

imagenette.to_csv('sup.csv', sep=',', index=False)
