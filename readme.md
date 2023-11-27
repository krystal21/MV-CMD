# Code for paper "Enhancing Nasopharyngeal Carcinoma Classification Based on Multi-View Cross-Modal Knowledge Distillation"

Our paper implements a multi-view, cross-modal distillation algorithm. The algorithm implementation comprises two phases, involving pre-training of the teacher model (utilizing multi-view contrastive learning) and multi-view weighted distillation. 

## Usage

**Training the Net-T**

1. Supervised

   ```python
   python train.py --config config/config.yml
   ```

2. MV-SupCon

   ```python
   python viewcon.py --config config/config.yml
   ```

**Training the Net-S**

```python
python train_kd.py --config config/config.yml
```

## DataSet

```yml
- /dataset
    - /nbi
        - 0train.csv
        - 0val.csv
        ...
    - /wle
        - 0train.csv
        - 0val.csv
        ...
    - /viewcon
        - 0train.csv
        - 0val.csv
        ...     
```

 CSV file stores filenames and labels. 

