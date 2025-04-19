import torch

def obb_to_xywha(quad: torch.Tensor) -> torch.Tensor:
    """
    Convert quadrilateral boxes (8 coords) to rotated boxes: [cx, cy, w, h, angle]

    Args:
        quad (Tensor[N, 8]): Quadrilateral boxes as (x1, y1, x2, y2, x3, y3, x4, y4)

    Returns:
        Tensor[N, 5]: Rotated boxes as (cx, cy, w, h, angle in radians)
    """
    quad = quad.view(-1, 4, 2)  # [N, 4, 2]
    
    # Compute center
    cx = quad[:, :, 0].mean(dim=1)
    cy = quad[:, :, 1].mean(dim=1)

    # Vector from point 1 to point 2 defines angle
    edge1 = quad[:, 1] - quad[:, 0]
    angle = torch.atan2(edge1[:, 1], edge1[:, 0])  # radians

    # Width = length of edge1
    w = edge1.norm(dim=1)

    # Height = length of edge2 (pt2 → pt3)
    edge2 = quad[:, 2] - quad[:, 1]
    h = edge2.norm(dim=1)

    return torch.stack([cx, cy, w, h, angle], dim=1)