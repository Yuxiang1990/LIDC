__author__ = "jiancheng"

import os
import glob
import json
import warnings

from bs4 import BeautifulSoup
import dicom
import numpy as np
from matplotlib.path import Path
import pandas as pd


class PathParser:

    def __init__(self, folder):
        self.folder = folder
        self.target_folder = self._select_target_sub_folder(
            glob.glob(folder + "/*"))
        self.xml_path = self._get_xml_path()
        self.dcm_paths = self._get_dcm_paths()

    def _select_target_sub_folder(self, subfolders):
        file_count = [len(glob.glob(p + '/*/*.dcm')) for p in subfolders]
        target = max(zip(file_count, subfolders))
        return target[1]

    def _get_xml_path(self):
        xml_path = glob.glob(self.target_folder + "/*/*.xml")
        assert len(xml_path) == 1, self.folder
        return xml_path[0]

    def _get_dcm_paths(self):
        dcm_paths = glob.glob(self.target_folder + "/*/*.dcm")
        assert dcm_paths, self.folder
        return dcm_paths


class XMLParser:

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.soup = BeautifulSoup(f.read(), "xml")
        self.parsed_dict = self._parse()

    def pprint(self):
        print(self.soup.prettify())

    def _parse(self):
        self.ret = {}
        readingSessions = self.soup.find_all('readingSession')
        assert len(readingSessions) == 4, self.path
        return {'readingSession':
                [self._parse_unblindedReadNodule(sess.find_all('unblindedReadNodule'))
                 for sess in readingSessions]}

    def _parse_unblindedReadNodule(self, unblindedReadNodules):
        nodule_large_than_3mm = [
            node for node in unblindedReadNodules if node.find('characteristics')]
        if not nodule_large_than_3mm:
            warnings.warn("no large nodule for " + self.path)
        return {"large_nodules": [{"malignancy": self._parse_malignancy(nodule),
                                   "roi": [self._parse_roi(roi) for roi in nodule.find_all('roi')]}
                                  for nodule in nodule_large_than_3mm]}

    def _parse_malignancy(self, nodule):
        return int(nodule.find('characteristics').find('malignancy').string)

    def _parse_roi(self, roi):
        imageZposition = float(roi.find('imageZposition').string)
        xCoords = [int(x.string) for x in roi.find_all('xCoord')]
        yCoords = [int(x.string) for x in roi.find_all('yCoord')]
        assert len(xCoords) == len(yCoords), self.path
        return {"imageZposition": imageZposition,
                "coords": list(zip(xCoords, yCoords))}

    def dump(self, filename):
        with open(filename, 'w') as f:
            f.write(self.dumps())

    def dumps(self):
        return json.dumps(self.parsed_dict, indent=4)


class Patient:

    def __init__(self, folder):
        self.path_parser = PathParser(folder)
        self.xml_parser = XMLParser(self.path_parser.xml_path)
        self.info = self.xml_parser.parsed_dict
        dss = [dicom.read_file(f) for f in self.path_parser.dcm_paths]
        self._process_ds(dss)
        del dss
        self.answer, self.concern = \
            get_person_answer(self.voxel.shape, self.info,
                              self.depth, silence=True)

    def _process_ds(self, dss):
        depth = [float(ds[0x20, 0x32].value[-1]) for ds in dss]
        self.depth = np.array(sorted(depth))
        pixel_array = sorted(zip(depth, [ds.pixel_array for ds in dss]))
        self.voxel = np.array([arr[1] for arr in pixel_array])
        assert self.voxel.shape[1] == self.voxel.shape[2] == 512, self.path_parser.folder
        del pixel_array

    def cache(self, name):
        with open('%s.npz' % name, 'wb') as f:
            np.savez_compressed(f, voxel=self.voxel,
                                answer=self.answer,
                                depth=self.depth,
                                concern=self.concern)
        self.xml_parser.dump('%s.json' % name)


def get_person_answer(shape, info, depth, silence=False):
    '''Now this implement is quite slow.'''
    if not silence:
        warnings.warn("Now the `get_person_answer` implementation is slow.")
    nodule_index = set()
    answer_map = np.zeros(shape)
    for sess in info['readingSession']:
        for nodule in sess['large_nodules']:
            malignancy = nodule['malignancy']
            for roi in nodule['roi']:
                xy = roi['coords']
                nodule_path = Path(np.array(xy))
                depth_index = list(depth).index(roi['imageZposition'])
                nodule_index.add(depth_index)
                inside = [[nodule_path.contains_point(
                    (x + 1, y + 1)) for x in range(shape[2])] for y in range(shape[1])]
                for x, y in xy:
                    inside[y][x] = True
                answer_map[depth_index] += np.array(inside) * malignancy / 4
    # map [0,5] => [0,9]
    answer_map = np.ceil((answer_map + 0.1) / 1.03 * 2) - 1
    return answer_map.astype(np.int16), np.array(list(nodule_index))


class DataCase:
    compressed_path = '/data1/data/DataSets/LIDC-IDRI/compressed/'
    xmin, xmax = 0, 512
    ymin, ymax = 60, 485
    absort_ratio = 0.12
    # absort_top = 15
    # vmax = 4095 # now use the dynamic mapping
    # vmin = -1024 # there is not only -1024

    def __init__(self, voxel, answer, mask, proposal, keep_original=False):
        if keep_original:
            self._voxel = voxel
            self._answer = answer
            self._mask = mask
            self._proposal = proposal

        assert voxel.shape == answer.shape == mask.shape
        self.absort_top = int(voxel.shape[0] * self.absort_ratio)
        self.absort_bottom = int(voxel.shape[0] * self.absort_ratio * 0.6)

        self.vmax = voxel.max()
        self.vmin = voxel.min()
        voxel = (voxel - self.vmin) / (self.vmax - self.vmin)
        voxel = voxel * mask
        voxel = np.expand_dims(voxel, axis=-1)
        self.voxel = voxel[self.absort_top:-self.absort_bottom,
                           self.ymin:self.ymax, self.xmin:self.xmax, :]
        answer = np.expand_dims(answer, axis=-1)
        self.answer = answer[self.absort_top:-self.absort_bottom,
                             self.ymin:self.ymax, self.xmin:self.xmax, :]
        mask = np.expand_dims(mask, axis=-1)
        self.mask = mask[self.absort_top:-self.absort_bottom,
                         self.ymin:self.ymax, self.xmin:self.xmax, :]
        self.proposal = proposal[self.absort_top:-self.absort_bottom,
                                 self.ymin:self.ymax, self.xmin:self.xmax]
        if not keep_original:
            del voxel, answer, mask, proposal

    @classmethod
    def from_cache(cls, name, keep_original=False):
        p = np.load(cls.compressed_path + name + ".npz")
        mask = np.load(cls.compressed_path + name + "_mask.npz")
        region = np.load(cls.compressed_path + name + "_region.npz")
        return cls(p['voxel'], p['answer'], mask['mask'], region['proposal'], keep_original)


def get_patient_info_df(ppath):
    name = "p%s" % int(ppath[-5:-1] if ppath.endswith('/')
                       else ppath[-4:])
    parser = PathParser(ppath)
    dcm_paths = parser.dcm_paths
    df = pd.DataFrame()
    df['path'] = dcm_paths
    df['name'] = name
    dcms = df['path'].map(lambda p: dicom.read_file(p))
    pixels = dcms.map(lambda dcm: dcm.pixel_array)
    df['array_max'] = pixels.map(lambda pixel: pixel.max())
    df['array_min'] = pixels.map(lambda pixel: pixel.min())
    df['intercept'] = dcms.map(lambda dcm: int(dcm.RescaleIntercept))
    df['slope'] = dcms.map(lambda dcm: int(dcm.RescaleSlope))
    df['slice_loc'] = dcms.map(lambda dcm: float(dcm.SliceLocation))
    df['z'] = dcms.map(lambda dcm: float(dcm.ImagePositionPatient[2]))
    df['thickness'] = dcms.map(lambda dcm: float(dcm.SliceThickness))
    df['index'] = dcms.map(lambda dcm: int(dcm.InstanceNumber) - 1)
    return df.set_index('index').sort_index()
