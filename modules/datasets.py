import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split_path = args.split_path

        self.attr_path = os.path.join(os.path.dirname(self.split_path), 'target_attributes.csv')

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        self.attr_cols = ['Grade_High', 'Multifocality', 'Necrosis', 'LVI', 'Subtype_Ductal', 'LN_Pos', 'DCIS']

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.id_to_pt_file = {}
        if os.path.exists(self.image_dir):
            print(f"[Init] Scanning feature directory: {self.image_dir}")
            for f in os.listdir(self.image_dir):
                if f.endswith('.pt'):
                    parts = f.split('-')
                    if len(parts) >= 3:
                        short_id = "-".join(parts[:3])
                        self.id_to_pt_file[short_id] = f
            print(f"[Init] Found {len(self.id_to_pt_file)} .pt files.")
        else:
            print(f"Error: Image dir not found: {self.image_dir}")
        # -----------------------------------------------------------

        if os.path.exists(self.attr_path):
            self.attr_df = pd.read_csv(self.attr_path)
            self.attr_df['Short_ID'] = self.attr_df['Slide_ID'].apply(lambda x: "-".join(x.split("-")[:3]))
            self.attr_df = self.attr_df.set_index('Short_ID')
            self.attr_df = self.attr_df[~self.attr_df.index.duplicated(keep='first')]
        else:
            print(f"Warning: Attribute file not found at {self.attr_path}")
            self.attr_df = pd.DataFrame()

        self.full_df = pd.read_csv(self.split_path)
        valid_cases_series = self.full_df.loc[:, self.split].dropna()
        cases = self.clean_data(valid_cases_series)

        self.examples = []
        root = self.ann_path

        if not os.path.exists(root):
             print(f"Error: Ann path not found: {root}")

        dir_list = os.listdir(root)
        print(f"[Init] Processing {len(dir_list)} annotation folders...")

        for dir in dir_list:
            short_id = "-".join(dir.split('-')[:3])

            if short_id not in cases.keys():
                continue
            
            if short_id not in self.id_to_pt_file:
                # print(f"Skipping {short_id}: .pt file not found.")
                continue
            
            pt_filename = self.id_to_pt_file[short_id]
            image_path = os.path.join(self.image_dir, pt_filename)
            
            file_name = os.path.join(root, dir, 'annotation')
            if not os.path.exists(file_name):
                if os.path.exists(file_name + '.json'): file_name += '.json'
                elif os.path.exists(file_name + '.txt'): file_name += '.txt'
                else: continue

            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    try:
                        json_data = json.loads(content)
                        if isinstance(json_data, dict):
                            if 'report' in json_data: anno = json_data['report']
                            elif 'text' in json_data: anno = json_data['text']
                            else: anno = " ".join([str(v) for v in json_data.values() if isinstance(v, str)])
                        else:
                            anno = str(json_data)
                    except json.JSONDecodeError:
                        anno = content
            except Exception as e:
                print(f"Error reading annotation for {dir}: {e}")
                continue

            report_ids = tokenizer(anno)
            if len(report_ids) < self.max_seq_length:
                padding = [0] * (self.max_seq_length - len(report_ids))
                report_ids.extend(padding)

            attr_labels, attr_masks = self.get_clinical_attributes(short_id)

            self.examples.append({
                'id': short_id,
                'image_path': image_path,
                'report': anno,
                'split': self.split,
                'ids': report_ids,
                'mask': [1] * len(report_ids),
                'attr_labels': attr_labels,
                'attr_masks': attr_masks
            })
        print(f'The size of {self.split} dataset: {len(self.examples)}')

    def get_clinical_attributes(self, case_id):
        labels = []
        masks = []
        if case_id in self.attr_df.index:
            row = self.attr_df.loc[case_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
                
            for col in self.attr_cols:
                if col in row:
                    val = row[col]
                    if val == 1:
                        labels.append(1.0);
                        masks.append(1.0)
                    elif val == 0:
                        labels.append(0.0);
                        masks.append(1.0)
                    else:
                        labels.append(0.0);
                        masks.append(0.0)
                else:
                    labels.append(0.0);
                    masks.append(0.0)
        else:
            labels = [0.0] * len(self.attr_cols)
            masks = [0.0] * len(self.attr_cols)
        return labels, masks

    def __len__(self):
        return len(self.examples)

    def clean_data(self, data):
        cases = {}
        for idx in range(len(data)):
            case_name = data.iloc[idx]
            case_id = '-'.join(case_name.split('-')[:3]).split('.')[0]
            cases[case_id] = case_id
        return cases


class TcgaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        try:
            image = torch.load(image_path)
            
            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image.squeeze(0)

            image = image[:self.max_fea_length]
        except Exception as e:
            print(f"Error loading pt file {image_path}: {e}")
            image = torch.zeros(10, 512)

        report_ids = example['ids']
        report_masks = example['mask']
        attr_labels = torch.FloatTensor(example['attr_labels'])
        attr_masks = torch.FloatTensor(example['attr_masks'])
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length, attr_labels, attr_masks)
        return sample

    def filter_df(self, df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self, data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data