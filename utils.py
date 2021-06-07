from django.conf import settings
from rest_framework.views import exception_handler
import requests
import numpy as np
from osgeo import gdal, ogr, osr
import os
from copernicus.consts import COPERNICUS_POLLUTANT_UNITS_TO_MULTIPLY, COPERNICUS_POLLUTANT_NAMES, COPERNICUS_LEVEL_TYPE, COPERNICUS_FILE_TYPE, COPERNICUS_AQI_FILE_TYPE
import glob
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from netCDF4 import Dataset
import numba as nb
from volumes.countries.countries_list import COUNTRIES_LIST
import subprocess
import tempfile
import datetime
import shutil
import re
import abc
import datetime
from .const import bbox_by_iso
from app_client.models import CountryBbox as Cbbx
from django.contrib.gis.geos import  Point
from .const import bbox_by_iso
import hashlib
import glob


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is not None:
        data = {
            "name": "ValidationError",
            "message": str(exc.detail),
            "code": exc.status_code,
            "status": 0,
            "type": exc.default_code
        }
        response.data = data

    return response


def get_airvisual_aqi(lon, lat):
    url = f"http://api.airvisual.com/v2/nearest_city?lat={lat}&lon={lon}&key={settings.AIRVISUAL_KEY}"
    resp = requests.get(url)
    if resp.status_code is 200:
        resp = resp.json()
        aqi = resp["data"]["current"]["pollution"].get("aqius")
        aqi_data = {
            "value": aqi,
            "status": get_status_aqi(aqi),
            "dominant_pollutant": resp["data"]["current"]["pollution"].get("mainus")
        }
        return aqi_data


def convert_ppm_to_m3(value: float, gas: str) -> float:
    if gas == "so2" or gas == "SO2":
        weight = 2.62
    elif gas == "no2" or gas == "NO2":
        weight = 1.88
    elif gas == "co" or gas == "CO":
        weight = 1.145
    elif gas == "o3" or gas == "OZONE":
        weight = 2.0
    else:
        raise ValueError("gas weight not found")
    return value * weight


def convert_to_x_y(lon ,lat):
    inp = osr.SpatialReference()
    inp.ImportFromEPSG(4326)
    out = osr.SpatialReference()
    out.ImportFromEPSG(3857)
    transformation = osr.CoordinateTransformation(inp, out)
    return transformation.TransformPoint(lon, lat)


def get_dataset_path(time_step, pollutant_name):
    file_path = 'z_cams_c_ecmf_{}_{}_{}_{}_{}_{}.{}'.format('*' + '0000', 'prod', 'fc', COPERNICUS_LEVEL_TYPE.get(
        pollutant_name), str(time_step).zfill(3), COPERNICUS_POLLUTANT_NAMES.get(pollutant_name), COPERNICUS_FILE_TYPE)
    file_search_path = os.path.join(settings.ECMWF_DATA_DIR, file_path)
    return glob.glob(file_search_path)[0]

def get_dataset_aqi_path(time_step, aqi_standard_id):
    file_path = '{}_{}.{}'.format(aqi_standard_id, str(time_step).zfill(3), COPERNICUS_AQI_FILE_TYPE)
    return os.path.join(settings.ECMWF_INDEX_DATA_DIR, file_path)

def get_dataset_interpolation_path(aqi_standard_id, creation_date=None):
    if creation_date:
        return build_dataset_interpolation_path(aqi_standard_id, creation_date)
    
    filenames = os.listdir(settings.INTERPOLATION_DATA_DIR)
    date_pattern = re.compile("(\d{10})\.")
    file_dates = [f[-13:-3] for f in filenames if date_pattern.search(f)]
    max_date = max(file_dates)
    return build_dataset_interpolation_path(aqi_standard_id, max_date)

def build_dataset_interpolation_path(aqi_standard_id, creation_date):
    file_name = f'{aqi_standard_id}{creation_date}.nc'
    file_path = os.path.join(settings.INTERPOLATION_DATA_DIR, file_name)
    return os.path.exists(file_path), file_path


def unit_to_multiply(pollutant_name):
    return COPERNICUS_POLLUTANT_UNITS_TO_MULTIPLY.get(pollutant_name)


class BaseInterpolation:

    def __init__(self, width, height, bbox):
        self.raster_width = width
        self.raster_height = height
        self.extent = self.get_extent(bbox)
        self.srs = self.get_spatial_reference()
        self.pixel_size = 10000
    
    def get_spatial_reference(self):
        spatialReference = osr.SpatialReference()
        spatialReference.ImportFromEPSG(4326)
        return spatialReference

    def get_extent(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        xmin, ymin, u = convert_to_x_y(x_min, y_min)
        xmax, ymax, u = convert_to_x_y(x_max, y_max)
        return xmin, ymin, xmax, ymax


    def get_resolution(self):
        x_min, y_min, x_max, y_max = self.extent
        x_res = int((x_max - x_min) / self.pixel_size)
        y_res = int((y_max - y_min) / self.pixel_size)
        return x_res, y_res


class Interpolation(BaseInterpolation):

    def __init__(self, width, height, bbox, data, algorithm, aqi_standard_id):
        self.source_ds = self.create_datasource(data)
        self.alg = algorithm
        self.aqi_standard_id = aqi_standard_id
        super().__init__(width, height, bbox)

    def create_datasource(self, data):
        driver = gdal.GetDriverByName('Memory')
        ds = driver.Create('', 0, 0, 0, gdal.GDT_Unknown)
        layer = ds.CreateLayer('memoryLayer', geom_type=ogr.wkbPoint)
        layer_defn = layer.GetLayerDefn()
        keys = ['aqi', 'x', 'y']
        for field in keys:
            # we will create a new field with the content of our header
            new_field = ogr.FieldDefn(field, ogr.OFTString)
            layer.CreateField(new_field)
        for item in data:
            point = ogr.Geometry(ogr.wkbPoint)
            # we do have LATs and LONs as Strings, so we convert them
            x, y, u = convert_to_x_y(float(item['station']['location']['coordinates'][0]), float(
                item['station']['location']['coordinates'][1]))
            point.AddPoint(x, y)
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(point)  # set the coordinates
            data_item = {'aqi': item['data']['aqi']['value'],
                         'x': x,
                         'y': y}
            for field in keys:
                i = feature.GetFieldIndex(field)
                feature.SetField(i, data_item[field])
            layer.CreateFeature(feature)
        return ds

    def interpolate(self):
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
        gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
        x, y = self.get_resolution()
        data = gdal.Grid('/vsimem/aqi_interp.tif', self.source_ds, layers='memoryLayer', zfield='aqi',
                         algorithm=self.alg,
                         outputType=gdal.GDT_Float64,
                         outputBounds=list(self.extent), outputSRS=self.srs, format='GTiff', width=self.raster_width, noData=0, height=self.raster_height)
        data_geo_transform = data.GetGeoTransform()
        if gdal.VSIStatL('/vsimem/aqi_interp.tif'):
            clipped_data = gdal.Warp('/vsimem/aqi_interp_cut.tif',
                                     data,
                                     cutlineDSName=os.path.join(
                                         settings.BASE_DIR, 'probe')+'/mask/',
                                     dstNodata=0)
            clipped_data.SetGeoTransform(data_geo_transform)
            result = gdal.DEMProcessing('/vsimem/aqi_interp_colored.png', clipped_data, processing='color-relief',
                                        colorFilename=os.path.join(settings.BASE_DIR, 'colors', f'{self.aqi_standard_id}.txt'), addAlpha=True)
            if gdal.VSIStatL('/vsimem/aqi_interp_colored.png'):
                f = gdal.VSIFOpenL('/vsimem/aqi_interp_colored.png', 'rb')
                gdal.VSIFSeekL(f, 0, 2)  # seek to end
                size = gdal.VSIFTellL(f)
                gdal.VSIFSeekL(f, 0, 0)  # seek to beginning
                data = gdal.VSIFReadL(1, size, f)
                gdal.VSIFCloseL(f)
                return data


class CopernicusInterpolation(BaseInterpolation):

    def __init__(self, width, height, bbox, ds_path, pollutant_name):
        self.ds_path = ds_path
        self.pollutant_name = pollutant_name
        super().__init__(width, height, bbox)

    def get_spatial_reference(self):
        spatialReference = osr.SpatialReference()
        spatialReference.ImportFromEPSG(3857)
        return spatialReference

    def interpolate(self):
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
        ds = gdal.Open(self.ds_path)
        if self.pollutant_name != 'pm25' and self.pollutant_name != 'pm10':
            ds = gdal.Translate('/vsimem/translated.tif', ds, width=self.raster_width, height=self.raster_height, bandList=[25])
        projected_ds = gdal.Warp('/vsimem/projected.tif', ds, width=self.raster_width, outputBoundsSRS=self.get_spatial_reference(
        ), outputBounds=list(self.extent), height=self.raster_height, dstSRS=self.get_spatial_reference(), resampleAlg='cubic')
        band = projected_ds.GetRasterBand(1).ReadAsArray()
        converted_band = unit_to_multiply(self.pollutant_name) * band
        projected_ds.GetRasterBand(1).WriteArray(converted_band)
        if gdal.VSIStatL('/vsimem/projected.tif'):
            result = gdal.DEMProcessing('/vsimem/copernicus_interp_colored.png', projected_ds, processing='color-relief',
                                        colorFilename=os.path.join(settings.BASE_DIR, 'colors', 'copernicus', f'{self.pollutant_name}.txt'), addAlpha=True)
            if gdal.VSIStatL('/vsimem/copernicus_interp_colored.png'):
                f = gdal.VSIFOpenL('/vsimem/copernicus_interp_colored.png', 'rb')
                gdal.VSIFSeekL(f, 0, 2)  # seek to end
                size = gdal.VSIFTellL(f)
                gdal.VSIFSeekL(f, 0, 0)  # seek to beginning
                data = gdal.VSIFReadL(1, size, f)
                gdal.VSIFCloseL(f)
                return data


class CopernicusAqiInterpolation(BaseInterpolation):

    def __init__(self, width, height, bbox, ds_path, aqi_standard_id):
        self.ds_path = ds_path
        self.aqi_standard_id = aqi_standard_id
        self.origin_bbox = bbox
        super().__init__(width, height, bbox)

    def get_spatial_reference(self):
        spatialReference = osr.SpatialReference()
        spatialReference.ImportFromEPSG(3857)
        return spatialReference

    def interpolate(self):
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
        ds = gdal.Open('NETCDF:"'+self.ds_path+'":aqi')
        projected_ds = gdal.Warp('/vsimem/projected_aqi.tif', ds, width=self.raster_width, dstSRS=self.get_spatial_reference(),outputBounds=list(self.origin_bbox), 
            height=self.raster_height, resampleAlg='cubic')
        if gdal.VSIStatL('/vsimem/projected_aqi.tif'):
            result = gdal.DEMProcessing('/vsimem/copernicus_aqi_interp_colored.png', projected_ds, processing='color-relief',
                                        colorFilename=os.path.join(settings.BASE_DIR, 'colors', f'{self.aqi_standard_id}.txt'), addAlpha=True)
            if gdal.VSIStatL('/vsimem/copernicus_aqi_interp_colored.png'):
                f = gdal.VSIFOpenL('/vsimem/copernicus_aqi_interp_colored.png', 'rb')
                gdal.VSIFSeekL(f, 0, 2)  # seek to end
                size = gdal.VSIFTellL(f)
                gdal.VSIFSeekL(f, 0, 0)  # seek to beginning
                data = gdal.VSIFReadL(1, size, f)
                gdal.VSIFCloseL(f)
                return data


class IdwInterpolation(BaseInterpolation):
    COUNTRIES_PATH = os.path.join(settings.BASE_DIR, 'volumes', 'countries')
    COUNTRIES_SHAPE_FILENAME = 'World_Countries__Generalized_'

    def __init__(self, width, height, bbox, ds_path, aqi_standard_id):
        self.ds_path = ds_path
        self.aqi_standard_id = aqi_standard_id
        super().__init__(width, height, bbox)

    def countries_sql_query(self):
        return f'select * from {self.COUNTRIES_SHAPE_FILENAME} where ISO IN {tuple(COUNTRIES_LIST)}'

    def interpolate(self):
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
        ds = gdal.Open('NETCDF:"'+self.ds_path+'":aqi')

        data = gdal.Warp('/vsimem/projected_interp_aqi.tif', ds, width=self.raster_width, dstSRS=self.get_spatial_reference(), outputBounds=self.extent, height=self.raster_height,  resampleAlg='cubic')
        if gdal.VSIStatL('/vsimem/projected_interp_aqi.tif'):
            clipped_data = gdal.Warp('/vsimem/aqi_interp_cut.tif',
                                     data,
                                     cutlineDSName=self.COUNTRIES_PATH,
                                     cutlineSQL=self.countries_sql_query(),
                                     dstNodata=0)
            result = gdal.DEMProcessing('/vsimem/aqi_interp_colored.png', clipped_data, processing='color-relief',
                                        colorFilename=os.path.join(settings.BASE_DIR, 'colors', f'{self.aqi_standard_id}.txt'), addAlpha=True)
            if gdal.VSIStatL('/vsimem/aqi_interp_colored.png'):
                f = gdal.VSIFOpenL('/vsimem/aqi_interp_colored.png', 'rb')
                gdal.VSIFSeekL(f, 0, 2)  # seek to end
                size = gdal.VSIFTellL(f)
                gdal.VSIFSeekL(f, 0, 0)  # seek to beginning
                data = gdal.VSIFReadL(1, size, f)
                gdal.VSIFCloseL(f)
                return data
    
    
    @staticmethod           
    def check_no_data_countries(lon, lat):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(IdwInterpolation.COUNTRIES_PATH, 0)
        layer = dataSource.GetLayer()
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(lon, lat)
        layer.SetSpatialFilter(point)
        feature = layer.GetNextFeature()
        if feature and feature.GetField('ISO') in COUNTRIES_LIST:
            return True
        return False
    
    
    def generate_tiles(self, processes, max_zoom, min_zoom, pixel_size):
        self.pixel_size = pixel_size
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
        gdal.SetConfigOption("GDAL_NUM_THREADS", "4")
        gdal.SetConfigOption("CPL_DEBUG", "ON")

        x, y = self.get_resolution()
        with tempfile.TemporaryDirectory() as temp_dir_name:
            projected_tif_path = os.path.join(temp_dir_name, 'projected_interp_aqi.tif')
            clipped_tif_path = os.path.join(temp_dir_name, 'aqi_interp_cut.tif')
            colored_png_path = os.path.join(temp_dir_name, 'aqi_interp_colored_masked.png')
            colors_path = os.path.join(settings.BASE_DIR, 'colors', f'{self.aqi_standard_id}.txt')
            subprocess.run(['gdalwarp', '-t_srs', 'EPSG:3857', '-ts', str(x), str(y), '-r',
                            'cubic', 'NETCDF:"'+self.ds_path+'":aqi', projected_tif_path], check=True)
            subprocess.run(['gdalwarp', '-dstnodata', '0', '-wm', '1000', '-multi', '-cutline', self.COUNTRIES_PATH,
                            '-csql', self.countries_sql_query(), projected_tif_path, clipped_tif_path], check=True)
            subprocess.run(['gdaldem', 'color-relief', clipped_tif_path,
                            colors_path, colored_png_path, '-alpha'], check=True)
            self.tiles_folder_name = self.aqi_standard_id + datetime.datetime.utcnow().strftime('%Y%m%d%H')
            self.tiles_date = datetime.datetime.utcnow()
            self.tiles_folder_path = os.path.join(settings.INTERPOLATION_DATA_TILES_DIR, self.tiles_folder_name)
            subprocess.run(['gdal2tiles.py', f'--zoom={min_zoom}-{max_zoom}', f'--processes={processes}',
                            '--webviewer=none', colored_png_path, self.tiles_folder_path], check=True)

@nb.njit(fastmath=True)
def haversine(lon1, lat1, lon2, lat2):

    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


@nb.njit(fastmath=True)
def get_idw_point(unknowncell_x, unknowncell_y, knowncells, power):
    numerator = 0
    denumerator = 0
    for knowncell in knowncells:
        distance = haversine(unknowncell_x, unknowncell_y, knowncell[0], knowncell[1])
        if distance == 0:
            distance = knowncell[2]
        numerator += knowncell[2] / distance ** power
        denumerator += 1 / distance ** power
    return numerator / denumerator


@nb.njit(parallel=True)
def interpolate(lons,lats,xx,yy,interpolated_values_grid, data, power):
    for lat in nb.prange(len(lats)):
        for lon in nb.prange(len(lons)):
            interpolated_values_grid[lat][lon] = get_idw_point(xx[lat][lon], yy[lat][lon], data, power)


def write_to_file(filename, lons, lats, interpolated_values_grid):
    ncout = Dataset(filename, 'w', format='NETCDF4')
    ncout.createDimension('lon', len(lons))
    ncout.createDimension('lat', len(lats))
    lonvar = ncout.createVariable('lon', 'float32', ('lon'))
    lonvar[:]=lons
    latvar = ncout.createVariable('lat', 'float32', ('lat'))
    latvar[:]=lats
    aqi = ncout.createVariable('aqi','float32',('lat','lon'))
    
    crs = ncout.createVariable('crs','int')
    crs.grid_mapping_name = "latitude_longitude"

    ncout.Conventions='CF-1.6'
    latvar.units = "degrees_north"
    latvar.long_name="latitude"
    lonvar.units = "degrees_east"
    lonvar.long_name="longitude"
    aqi[:] = interpolated_values_grid
    aqi.grid_mapping = "crs"
    ncout.close()


def idw(data, grid_cell_size ,power=2):
    lat_len = int((180 / grid_cell_size)+1)
    lon_len = int(360 / grid_cell_size)
    lons = np.linspace(-180, 180, lon_len, dtype=np.float32)
    lats = np.linspace(90, -90, lat_len, dtype=np.float32)
    xx, yy = np.meshgrid(lons, lats)
    interpolated_values_grid = np.zeros((len(lats), len(lons)), dtype=np.float32)
    interpolate(lons,lats,xx,yy,interpolated_values_grid,np.array(data, dtype=np.float32),power)
    return lons, lats, interpolated_values_grid

class Storage:


    def __init__(self, tiles_date, aqi_standard_id, days_keep_period=7):
        self.tiles_date = tiles_date
        self.days_keep_period = days_keep_period
        self.aqi_standard_id = aqi_standard_id
        self.name_to_remove = self.get_old_name()


    def get_old_name(self):
        date_old = self.tiles_date - datetime.timedelta(days=self.days_keep_period)
        return f"{self.aqi_standard_id}{date_old.strftime('%Y%m%d%H')}"


    def upload(self):
        pass


    def check_existence(self, path):
        pass


    def remove(self):
        pass



class AwsStorage(Storage):


    def __init__(self, bucket_name, tiles_folder_name, tiles_folder_path, tiles_date, aqi_standard_id, days_keep_period=7):
        self.bucket_name = bucket_name
        self.tiles_folder_name = tiles_folder_name
        self.tiles_folder_path = tiles_folder_path
        super().__init__(tiles_date, aqi_standard_id, days_keep_period)


    def upload(self):
        bucket_path = f"{self.bucket_name}/{self.tiles_folder_name}"
        args = ['aws', 's3', 'cp', self.tiles_folder_path, bucket_path, '--recursive']
        result = subprocess.run(args)
        if self.check_existence(bucket_path):
            return True
    
    def check_existence(self, bucket_path):
        ls_args = ['aws', 's3', 'ls', bucket_path]
        result = subprocess.run(ls_args)
        if result:
            return True

    def remove(self):
        old_bucket_folder_name = f"{self.bucket_name}/{self.name_to_remove}"
        if self.check_existence(old_bucket_folder_name):
            shutil.rmtree(self.tiles_folder_path)
            args = ['aws', 's3', 'rm', old_bucket_folder_name, '--recursive']
            subprocess.run(args)


    
class LocalStorage(Storage):


    def __init__(self, interpolation_file_name, interpolation_file_path, tiles_date, aqi_standard_id, days_keep_period=7):
        self.interpolation_file_name = interpolation_file_name
        self.interpolation_file_path = interpolation_file_path
        super().__init__(tiles_date, aqi_standard_id, days_keep_period)


    def check_existence(self, path):
        if os.path.exists(path):
            return True


    def get_old_name(self):
        date_old = self.tiles_date - datetime.timedelta(days=self.days_keep_period)
        return f"{self.aqi_standard_id}{date_old.strftime('%Y%m%d%H')}.nc"


    def remove(self):
        file_to_remove = os.path.join(self.interpolation_file_path, self.name_to_remove)
        if self.check_existence(file_to_remove):
            os.remove(file_to_remove)


class CountryBbox(abc.ABC):
    @abc.abstractclassmethod
    def get_country_bbox(self):
        pass


class CountryBboxByPoint(CountryBbox):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_country_bbox(self):
        try:
            point=Point(self.y,self.x)
            country=Cbbx.objects.filter(polygone__intersects=point).all().values('iso_code')            
            return bbox_by_iso[country[0]['iso_code']]
        except (IndexError, Exception):
            return None


class CountryBboxByIso(CountryBbox):

    def __init__(self, iso_code):
        self.iso_code = iso_code

    def get_country_bbox(self):
        if self.iso_code:
            return  bbox_by_iso[self.iso_code]
        return None

