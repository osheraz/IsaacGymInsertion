from isaacgyminsertion.utils.torch_jit_utils import *
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R

def axis_angle2quat(axis_angle_error):
    angle = torch.norm(-axis_angle_error, p=2, dim=-1)
    axis = -axis_angle_error / angle.unsqueeze(-1)
    quat = quat_from_angle_axis(angle, axis)
    return quat

def matrix_to_euler_xyz(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles in the XYZ convention.
    Args:
        rotation_matrix: tensor of shape (..., 3, 3).
    Returns:
        Euler angles as tensor of shape (..., 3).
    """
    # Ensure the input matrix is of shape (..., 3, 3)
    assert rotation_matrix.shape[-2:] == (3, 3), "Invalid shape of rotation matrix"

    # Extract matrix components for readability
    r11 = rotation_matrix[..., 0, 0]
    r21 = rotation_matrix[..., 1, 0]
    r31 = rotation_matrix[..., 2, 0]
    r32 = rotation_matrix[..., 2, 1]
    r33 = rotation_matrix[..., 2, 2]

    # Compute Euler angles
    theta_y = torch.asin(-r31)
    theta_z = torch.atan2(r32, r33)
    theta_x = torch.atan2(r21, r11)

    return torch.stack((theta_x, theta_y, theta_z), -1)


def quat2euler(quat):
    euler = get_euler_xyz(quat)
    euler_x, euler_y, euler_z = euler
    euler = torch.stack([euler_x, euler_y, euler_z], dim=-1)
    return euler

def quat2euler2(quat):
    M = quaternion_to_matrix(quat)
    euler = matrix_to_euler_xyz(M)
    return euler



def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[0] @ matrices[1] @ matrices[2]


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def xyzquat_to_tf_numpy(position_quat: np.ndarray) -> np.ndarray:
    """
    convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
    """
    # try:
    position_quat = np.atleast_2d(position_quat)  # (N, 7)
    N = position_quat.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
    T[:, :3, 3] = position_quat[:, :3]
    T[:, 3, 3] = 1
    # except ValueError:
    #     print("Zero quat error!")
    # return T.squeeze()
    return T

# pose vector: [position(3), quaternion(4)] = [x, y, z, q1, q2, q3, q0]
# pose matrix (SE3): [[R, p], [0^T, 1]]
# (7, ) or (N, 7) -> (4, 4) or (N, 4, 4)
def pose_vec_to_mat(input_vec: torch.Tensor):

    if isinstance(input_vec, np.ndarray):
        vec = torch.Tensor(input_vec)
    else:
        vec = input_vec

    assert vec.ndim in {1, 2} and vec.size(-1) == 7, f"invalid pose vector shape: {vec.size()}"
    ndim = vec.ndim
    if ndim == 1: vec = vec.view((1, -1)) # (1, 7)
    N = vec.size(0)

    p = vec[..., :3].view((-1, 3, 1))
    quat = vec[..., 3:]
    R = quat2R(quat)

    SE3 = torch.cat((
        torch.cat((R, p), dim=-1),
        torch.cat((torch.zeros((N, 1, 3), device=vec.device), torch.ones((N, 1, 1), device=vec.device)), dim=-1)
    ), dim=-2)

    if ndim == 1: SE3 = SE3.squeeze()

    # if isinstance(input_vec, np.ndarray):
    #     SE3 = SE3.cpu().numpy()

    return SE3

# (unit) quaternion to rotation matrix
# [vec, w] = [q1, q2, q3, q0] -> R
# (4, ) or (N, 4)-> (3, 3) or (N, 3, 3)
def quat2R(quat: torch.Tensor):
    assert quat.ndim in {1, 2} and quat.size(-1) == 4, f"invalid quaternion shape: {quat.size()}"
    quat = unify(quat)
    ndim = quat.ndim
    if ndim == 1: quat = quat.view((1, -1)) # (1, 4)

    q0, q1, q2, q3 = quat[..., -1], quat[..., 0], quat[..., 1], quat[..., 2] # (N, )
    # print([q.size() for q in [q0, q1, q2, q3]])
    R = torch.stack([
        torch.stack([1-2*q2**2-2*q3**2,   2*q1*q2-2*q0*q3,   2*q1*q3+2*q0*q2], dim=-1),
        torch.stack([  2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2,   2*q2*q3-2*q0*q1], dim=-1),
        torch.stack([  2*q1*q3-2*q0*q2,   2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2], dim=-1)
    ], dim=1) # (N, 3, 3)

    if ndim == 1: R = R.squeeze()

    return R

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def unify(x, eps: float = 1e-9): # (dim, ) or (N, dim)
    return x / torch.clamp(torch.linalg.norm(x, axis=-1, keepdims=True), min=eps, max=None)
