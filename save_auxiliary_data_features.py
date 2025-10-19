import fire
import os
import torch
from data import SyntheticDataset, get_loader
import models
import utils as u
from tqdm import tqdm

def save_features(train_loader, model, feature_path, debug):
    if torch.cuda.device_count() == 1:
        model.cuda()
    else:
        print('>1 GPUs detected, keeping model device unchanged since HF might have automatically sharded the model')
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
                if 'labels' in inputs:
                    inputs.pop('labels')
            if hasattr(inputs, 'items'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = inputs.cuda()
            feat = model(inputs).detach().cpu()
            features.append(feat)
            if debug:
                break
        features = torch.cat(features, dim=0)
    print(f'Features: {features.size()}')
    feature_dict = {'train': features}
    if not debug:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(feature_dict, feature_path)
        print(f'Saved features to {feature_path}')

def main(model_class, directory, class_file=None, num_images=None, batch_size=128, num_workers=0, save_path=None, debug=False, **kwargs):
    print("Saving auxiliary data features...")
    assert save_path is not None, "Please specify a save_path"
    args = locals()
    u.pretty_print_dict(args)
    model, get_transform, tokenizer, input_collate_fn = models.create_model(model_class, out_dim=0, pretrained=True, extract_features=True, **kwargs)
    model.eval()
    synthetic_ds = SyntheticDataset(directory, class_file, num_images, transform=get_transform(train=True))
    train_loader = get_loader(synthetic_ds, batch_size, num_workers=num_workers, shuffle=False, input_collate_fn=input_collate_fn)
    save_features(train_loader, model, save_path, debug)

if __name__ == '__main__':
    fire.Fire(main)
