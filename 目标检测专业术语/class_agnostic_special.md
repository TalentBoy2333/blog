class-aware detector：检测出的Bounding Box具有类别（猫，狗，飞机，汽车）



class-agnostic detector：只检测出“前景”目标Bounding Box



在做bounding box回归时，对bounding box的类别关注与否。

Class agnostic:

```Python
RCNN_bbox_pred = nn.Linear(2048, 4)
```

Class special:

```Python
RCNN_bbox_pred = nn.Linear(2048, 4 * n_classes)
```



For a class-aware detector, if you feed it an image, it will return a set of bounding boxes, each box associated with the class of the object inside (i.e. dog, cat, car). It means that by the time the detector finished detecting, it knows what type of object was detected.

For class-agnostic detector, it detects a bunch of objects without knowing what class they belong to. To put it simply, they only detect “foreground” objects. Foreground is a broad term, but usually it is a set that contains all specific classes we want to find in an image, i.e. foreground = {cat, dog, car, airplane, …}. Since it doesn’t know the class of the object it detected, we call it class-agnostic.

Class-agnostic detectors are often used as a pre-processor: to produce a bunch of interesting bounding boxes that have a high chance of containing cat, dog, car, etc. Obviously, we need a specialized classifier after a class-agnostic detector to actually know what class each bounding box contains.