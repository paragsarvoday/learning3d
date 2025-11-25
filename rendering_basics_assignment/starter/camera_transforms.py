"""
Something is very off with the relative rotation and translation, I will tackle it later. Bye!
"""

"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""

"""
Documentation:
To get the first picture:
    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, np.pi/2]), "XYZ")
    relative_translation = torch.tensor([0, 0, 0])

To get the second picture:
    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ")
    relative_translation = torch.tensor([0, 0, 3])

To get the third picture:
    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ")
    relative_translation = torch.tensor([0.5, 0, 0])
"""

import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer
# from pytorch3d.renderer import look_at_view_transform


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    if not torch.is_tensor(R_relative):
        R_relative = torch.tensor(R_relative).float()
    if not torch.is_tensor(T_relative):
        T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow.jpg")
    args = parser.parse_args()

    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
    torch.tensor([0, 0, 0]), "XYZ")

    # print(relative_rotation)

    from scipy.spatial.transform import Rotation as _R
    # r = _R.from_rotvec(np.pi/2 * np.array([0, 0, -1]))
    # R_rel = torch.from_numpy(r.as_matrix()).float()

    # relative_rotation = R_rel

    r = _R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
    R_rel = torch.from_numpy(r.as_matrix()).float()
    relative_rotation = R_rel
# T_rel = torch.tensor([-3, 0, 3]).float()

    relative_translation = torch.tensor([-3, 0, 3])
    img = render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative = relative_rotation, T_relative= relative_translation)
    plt.imsave(args.output_path, img)