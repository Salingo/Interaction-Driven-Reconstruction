import open3d as o3d
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
import os 
from skimage import measure


# 保存为OBJ文件
def save_obj(filename, vertices, faces):
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-dir', dest='input_dir', required=True)
    parser.add_argument('-n',type=int, dest='num_points', default=4096)
    parser.add_argument('-gs',type=int, dest='grid_size', default=16)
    args = parser.parse_args()
    
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    
    input_dir = args.input_dir
    grid_size = args.grid_size
    num_points = args.num_points
    
    out_dir = input_dir + f"_out_{num_points}_{grid_size}/{formatted_time}/"
    os.makedirs(out_dir)
    
    for pts_file in tqdm(os.listdir(input_dir)):
        point_cloud = o3d.geometry.PointCloud()
        points = np.loadtxt(f"{input_dir}/{pts_file}", dtype=np.float32)
        points = points[points[:,3]==1][:,:3]
        points = points[:,:3]
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_marching_cubes(point_cloud, voxel_size=voxel_size)
        file_name, _ = os.path.splitext(pts_file)

        # 将点云转换为体数据（体素化）
        grid, edges = np.histogramdd(points, bins=grid_size)

        # 使用Marching Cubes算法提取等值面
        vertices, faces, _, _ = measure.marching_cubes(grid, level=0.5)

        # 保存OBJ文件
        save_obj(f"{out_dir}/{file_name}.obj", vertices, faces)
        