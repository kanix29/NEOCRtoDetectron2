import numpy as np
import glob
import xmltodict
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from detectron2.structures import BoxMode

def get_neocr_dicts(xml_dir):
    xml_files = glob.glob(f'{xml_dir}/*.xml')
    xml_files.sort()

    dataset_dicts = []
    for idx, xml_file in enumerate(tqdm_notebook(xml_files)):
        # Load XML format to Dict
        doc = xmltodict.parse(open(xml_file).read())

        filename = os.path.join(img_dir, doc['annotation']['filename'])
        height, width = cv2.imread(filename).shape[:2]

        record = {}
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height # different from doc['annotation']['properties']['height']
        record["width"] = width # different from doc['annotation']['properties']['width']

        # for single object
        if not type(doc['annotation']['object']) == list:
            doc['annotation']['object'] = [doc['annotation']['object']]

        objs = []
        # Explore every object
        for ann_object in doc['annotation']['object']:

            # Get bbox of this object
            rectangle = []
            for pts in ann_object['polygon']['pt']:
                x, y = int(pts['x']), int(pts['y'])
                coordinate = [x, y]
                rectangle.append(coordinate)
            rectangle = np.array(rectangle)

            x_min, y_min = np.min(rectangle, axis=0)
            x_max, y_max = np.max(rectangle, axis=0)

            obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                    # Specify coordinates so that it goes around the boundary.
                    "segmentation": [[x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min]],
                }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts