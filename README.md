# System Requirements <br>   
<br>
Python 3.8.10        <br>    
RDKit 2022.09.5<br>
torch: 1.11.0+cu113<br>
dgl: 0.9.1.post1<br>
dgllife: 0.3.2<br>
numpy: 1.23.5<br>
scikit-learn: 1.1.3<br>
pandas: 1.5.3<br>
<br>
# Datasets <br>
<br>
All the datasets we use are publicly available.The 'datasets' folder of this project already contains all the required datasets.<br>
BindingDB can be obtained at https://www.bindingdb.org/bind/index.jsp;<br>
BioSNAP and initial version DAVIS datasets can be obtained at https://github.com/kexinhuang12345/MolTrans;<br>
Human source is at https://github.com/lifanchen-simm/transformerCPI.<br>
<br>
# Run DefuseDTI on Our Data to Reproduce Results <br>
<br>
python main.py --cfg "configs/DefuseDTI.yaml" --data "bindingdb" --split "random"<br>
python main.py --cfg "configs/DefuseDTI.yaml" --data "biosnap" --split "random"<br>
python main.py --cfg "configs/DefuseDTI.yaml" --data "human" --split "cold"<br>
python main.py --cfg "configs/DefuseDTI.yaml" --data "human" --split "random"<br>
python main.py --cfg "configs/DefuseDTI.yaml" --data "DAVIS" --split "random"<br>
<br>
Dear editors and reviewers, please feel free to contact us if there are any issues during the software execution！
 
 
