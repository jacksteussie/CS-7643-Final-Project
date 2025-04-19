import torch
from shapely.geometry import Polygon

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

def rotated_box_to_polygon(box):
    """
    Convert a rotated box [cx, cy, w, h, angle_degrees] to a shapely Polygon
    """
    cx, cy, w, h, angle = box.tolist()
    angle = angle * 180.0 / torch.pi  # rad to deg

    # points in box's local space
    dx = w / 2
    dy = h / 2
    corners = torch.tensor([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])

    # rotation matrix
    theta = angle * torch.pi / 180
    rot = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ])

    rotated = corners @ rot.T + torch.tensor([cx, cy])
    return Polygon(rotated.numpy())

def rotated_iou(box1, box2):
    """Compute IoU between two rotated boxes using shapely"""
    poly1 = rotated_box_to_polygon(box1)
    poly2 = rotated_box_to_polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def rotated_nms(boxes, scores, iou_threshold):
    """
    Pure Python rotated NMS using shapely.
    boxes: Tensor[N, 5] in (cx, cy, w, h, angle)
    scores: Tensor[N]
    """
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())
        if idxs.numel() == 1:
            break
        current_box = boxes[current]
        rest = idxs[1:]

        ious = torch.tensor([
            rotated_iou(current_box, boxes[i]) for i in rest
        ])

        idxs = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
