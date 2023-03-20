from modeling.utils.constants import MP3D_CATEGORIES


dict_cat2idx = {k: idx for idx, k in enumerate(MP3D_CATEGORIES)}
dict_idx2cat = {idx: v for idx, v in enumerate(MP3D_CATEGORIES)}

print(f'dict_cat2idx = {dict_cat2idx}')
print(f'dict_idx2cat = {dict_idx2cat}')
