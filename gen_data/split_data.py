import os 


data_dir = "/mnt/disk3/zihao/dev/pointnet_tem/data/score"

cate_list = ['dishwasher','kitchenpot','microwave','oven','refrige','storage','table','trashcan']

train_list = []
val_list = []

for cate in cate_list:
    data_map = set()
    for file_name in os.listdir(f"{data_dir}/{cate}"):
        shape_id, sid, pid, fid = file_name.split('_')
        data_map.add(f"{cate} {shape_id}")
        
    data_list = list(data_map)        
    len_data = len(data_list)
    len_train = int(len_data * 0.8)
    
    train_list.extend(data_list[:len_train])
    val_list.extend(data_list[len_train:])

print(f"train: {len(train_list)}")
print(f"val: {len(val_list)}")


with open(f"{data_dir}/train.txt", "w") as fp:
    for item in train_list:
        fp.write(item+"\n")
    
with open(f"{data_dir}/val.txt", "w") as fp:
    for item in val_list:
        fp.write(item+"\n")
        
print("Done!")
