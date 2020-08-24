from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import glob
from tqdm import tqdm

from detectron2.structures import BoxMode


def get_neocr_dicts(img_dir, xml_dir):
    ENCODE_METHOD = 'utf-8'
    num_files = len(os.listdir(xml_dir))

    dataset_dicts = []
    for idx, xml_file in tqdm(enumerate(glob.iglob(f'{xml_dir}/*.xml')), total=num_files):
        record = {}
        
        # process XML
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(xml_file, parser=parser).getroot()

        filename = os.path.join(img_dir, xmltree.find('filename').text)# ~.jpg
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        objs = []
        for object_iter in xmltree.findall('object'):

            polygon_iter = object_iter.find("polygon")

            rectangle = []
            for pt_iter in polygon_iter.findall("pt"):
                x = int(pt_iter.find('x').text)
                y = int(pt_iter.find('y').text)
                coordinate = [x, y]
                rectangle.append(coordinate)
            rectangle = np.array(rectangle)

            x_min, y_min = np.min(rectangle, axis=0)
            x_max, y_max = np.max(rectangle, axis=0)
            
            obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
            
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts