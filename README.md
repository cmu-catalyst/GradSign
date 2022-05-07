# GradSign
Code for paper [GradSign: Model Performance Inference with Theoretical Insights](https://openreview.net/pdf?id=HObMhrCeAAF)

### Spearman's rho experiments:
Clone and follow the installation steps in [zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas). Then type the following
command in the cmd: 

```
cp zero-cost-nas-code/*.py zero-cost-nas/
cp -r zero-cost-nas-code/foresight/pruners zero-cost-nas/foresight
cd zero-cost-nas/
python build.py
python gs.py

```

Modify `args` to test for different datasets.

### Kendall's Tau experiments:
Clone and follow the installation steps in [NASWOT](https://github.com/BayesWatch/nas-without-training). Then type the following
command in the cmd: 

```
cp naswot-code/*.py naswot/
cd naswot
python gradsign.py
```
Modify `args` to test for different datasets.

### Architecture selection experiments:
After finishing above steps from [NASWOT](https://github.com/BayesWatch/nas-without-training). Run:

```
python grad_search.py
```
Modify `args` to test for different datasets.

### GradSign assisted NAS:
Clone and follow the installation steps in [AutoDL](https://github.com/D-X-Y/AutoDL-Projects). Then type the following
command in the cmd: 
```
# Replace exps/NAS-Bench-201-algos/*.py with ours
cp autodl/*.py AutoDL-Projects/exps/NAS-Bench-201-algos/
```
Then simply run the command listed in [here](https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/NAS-Bench-201.md). The `GsApi` is built in the `zero-cost-nas` step and could be directly linked by changing
`gs_root`.
In order to run BOHB, you need to modify the source file of package `hpbandster` by replacing their `optimizers` 
directory with ours.
