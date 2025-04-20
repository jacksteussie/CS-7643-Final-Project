import torch

def norm_quad_to_aabb(quads, img_w, img_h):
    """
    Convert normalized quadrilateral coordinates [x1, y1, ..., x4, y4]
    to axis-aligned bounding boxes [x_min, y_min, x_max, y_max] in pixel coordinates.
    """
    quads = quads.view(-1, 4, 2)  # [N, 4, 2]
    quads[:, :, 0] *= img_w  # x
    quads[:, :, 1] *= img_h  # y
    x_min = torch.min(quads[:, :, 0], dim=1).values
    y_min = torch.min(quads[:, :, 1], dim=1).values
    x_max = torch.max(quads[:, :, 0], dim=1).values
    y_max = torch.max(quads[:, :, 1], dim=1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)  # [N, 4]