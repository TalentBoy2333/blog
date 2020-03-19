在做bounding box回归时，对bounding box的类别关注与否。

Class agnostic:

```Python
RCNN_bbox_pred = nn.Linear(2048, 4)
```

Class special:

```Python
RCNN_bbox_pred = nn.Linear(2048, 4 * n_classes)
```

