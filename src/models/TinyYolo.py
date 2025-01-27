import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Reusable convolutional block consisting of:
    Conv2d -> ReLU -> MaxPool2d -> BatchNorm2d
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.block(x)

class GateLayer(nn.Module):
    """
    Applies a 0/1 gate based on prob>0.5 to boxes and classes.
    """
    def __init__(self):
        super(GateLayer, self).__init__()

    def forward(self, x_prob, x_boxes, x_cls):
        gate = (x_prob > 0.5).float()
        x_boxes_gated = x_boxes * gate
        x_cls_gated = x_cls * gate
        return x_prob, x_boxes_gated, x_cls_gated

class TinyYOLO(nn.Module):
    """
    Tiny YOLO model for object detection.
    
    Architecture:
    - Input:  (B, 1, 128, 256) (B,1,H,W)
    - Output: (B, 7, 8, 8)
    
    The 4 max pools reduce:
        height: 128→64→32→16→8
        width:  256→128→64→32→16
    """
    def __init__(self):
        super(TinyYOLO, self).__init__()
        
        # Initial downsampling conv
        self.downconv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1)
        )
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16)
        )
        
        # Detection heads
        self.x_prob = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.x_boxes = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.x_cls = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        
        self.gate = GateLayer()

    def forward(self, x):
        # Initial processing
        x = self.downconv(x)
        
        # Feature extraction
        x = self.features(x)
        
        # Detection heads
        prob = torch.sigmoid(self.x_prob(x))
        boxes = self.x_boxes(x)
        cls_ = torch.sigmoid(self.x_cls(x))
        
        # Apply gating
        prob, boxes_gated, cls_gated = self.gate(prob, boxes, cls_)
        
        # Combine outputs
        out = torch.cat([prob, boxes_gated, cls_gated], dim=1)
        return out

class YOLOLoss(nn.Module):
    """
    Custom loss function for YOLO model combining:
    - Object/No-object confidence loss
    - Bounding box coordinate loss
    - Classification loss
    """
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord  # weight for coordinate loss
        self.lambda_obj = lambda_obj      # weight for object confidence loss
        self.lambda_noobj = lambda_noobj  # weight for no-object confidence loss
    
    def forward(self, y_pred, y_true, return_components=False):
        """
        Args:
            y_pred: (batch_size, 7, 8, 8) - predicted output
            y_true: (batch_size, 7, 8, 8) - ground truth
            return_components: if True, returns individual loss components
        """
        batch_size = y_pred.size(0)
        
        # Extract components from predictions and targets
        pred_conf = y_pred[:, 0]          # (batch,8,8)
        pred_xy   = y_pred[:, 1:3]        # (batch,2,8,8)
        pred_wh   = y_pred[:, 3:5]        # (batch,2,8,8)
        pred_cls  = y_pred[:, 5:7]        # (batch,2,8,8)
        
        targ_conf = y_true[:, 0]          # (batch,8,8)
        targ_xy   = y_true[:, 1:3]        # (batch,2,8,8)
        targ_wh   = y_true[:, 3:5]        # (batch,2,8,8)
        targ_cls  = y_true[:, 5:7]        # (batch,2,8,8)
        
        # Compute masks for objects/no-objects
        obj_mask = (targ_conf > 0).float()
        noobj_mask = (targ_conf == 0).float()
        
        # 1) Confidence loss (split into object/no-object)
        obj_loss = (obj_mask * (pred_conf - targ_conf).pow(2)).sum() / batch_size
        noobj_loss = (noobj_mask * (pred_conf - targ_conf).pow(2)).sum() / batch_size
        
        # 2) Coordinate loss (only for objects)
        coord_loss = (
            obj_mask.unsqueeze(1) * (
                (pred_xy - targ_xy).pow(2) +
                (pred_wh - targ_wh).pow(2)
            )
        ).sum() / batch_size
        
        # 3) Classification loss (only for objects)
        class_loss = (
            obj_mask.unsqueeze(1) * (pred_cls - targ_cls).pow(2)
        ).sum() / batch_size
        
        # Combine all losses with weights
        total_loss = (
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_coord * coord_loss +
            class_loss
        )
        
        if return_components:
            return total_loss, {
                'obj_loss': obj_loss.item(),
                'noobj_loss': noobj_loss.item(),
                'coord_loss': coord_loss.item(),
                'class_loss': class_loss.item()
            }
        return total_loss
