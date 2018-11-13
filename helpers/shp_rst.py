#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:38:01 2018

@author: eid
"""

import gdal, osr, ogr
import sys
import numpy as np

import rasterio
import rasterio.features
import json
import cv2
import os



def read_raster(rst_path, band_number):
    '''
    read a tif file and return the band with its attributes
    band_number is an integer starting with 1
    '''
    dataset = rasterio.open(rst_path)

    band  = dataset.read(band_number)
    ncol = dataset.width
    nrow = dataset.height
    nodata_val = dataset.nodata
    epsg = dataset.crs
    geoTransform = dataset.transform
    
    
   
    
    return (band,geoTransform,ncol,nrow,epsg,nodata_val)

def read_shpfile(in_shp):
    '''
    read shpfile projection and  attributes
    Returns GEOM (wkt format), EPSG (number), 
    FIELD DEFINATIONS(list containing the field data type, name and so..)
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # get the field  definition for the id  from the input shapefile
    shp = driver.Open(in_shp, 0) # 0 means read-only. 1 means writeable.
    
    layer = shp.GetLayer()
    
    # get Spatial Reference System (SRS) from Layer
    spatial_ref = layer.GetSpatialRef()
    srs = osr.SpatialReference(str(spatial_ref))
    res = srs.AutoIdentifyEPSG()
    if res == 0:
        srid = srs.GetAuthorityCode(None)
    else:
        print('Could not determine SRID')
    epsg = srid
     
    # get the name of the shpfile's field definition
    field_definition = []
    layerDefinition = layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        field_name =  layerDefinition.GetFieldDefn(i).GetName()
        field_type_code = layerDefinition.GetFieldDefn(i).GetType()
        field_type = layerDefinition.GetFieldDefn(i).GetFieldTypeName(field_type_code)
        field_width = layerDefinition.GetFieldDefn(i).GetWidth()
        field_precision = layerDefinition.GetFieldDefn(i).GetPrecision()
        field_definition.append([field_name, field_type_code, field_type, field_width, field_precision])
        
    # getting the GEOM and the attributes of the shpfiles
    data = []
    for feature in layer:
        out = []
        geom = feature.GetGeometryRef()
        geom_wkt = geom.ExportToJson()
        out.append(geom_wkt)
        
        for i in range(len(field_definition)):
            x = feature.GetField(i)
            out.append(x)
        data.append(out)
        
    return data, epsg, field_definition





def extract_field(rst_path,boundary,save_path = None):
    
    '''
    this function takes as input a raster file with boundary and cut the area surounded by that boundary
    '''
    
    
    #read each band from the raster
    band_R,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 1)
    band_G,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 2)
    band_B,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 3)
    
    #find the intersection area between the band and the raster
    boundary_as_raster_boundary = rasterio.features.geometry_mask([boundary], (ncol,nrow), geoTransform, all_touched=0, invert=1).astype(np.uint8)*255
    field_idxs = np.where(boundary_as_raster_boundary>0)
    
    
    #create the field image
    field = np.zeros((band_R.shape[0],band_R.shape[1],3),dtype = np.uint8)
    field [:,:,0][field_idxs] = band_B[field_idxs]
    field [:,:,1][field_idxs] = band_G[field_idxs]
    field [:,:,2][field_idxs] = band_R[field_idxs]
    
    if save_path != None:
        cv2.imwrite(save_path,field)
   
    return field 


from scipy.ndimage.morphology import binary_dilation
def draw_boundary(rst_path,boundary,save_path = None):
    
    '''
     this function draws the boundary around the field and save it as a jpg file
    '''
    
    #read each band from the raster
    band_R,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 1)
    band_G,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 2)
    band_B,geoTransform,ncol,nrow,epsg,nodata_val = read_raster(rst_path, 3)
    
    # find the intersection
    boundary_as_raster_boundary = rasterio.features.geometry_mask([boundary], (ncol,nrow), geoTransform, all_touched=0, invert=1).astype(np.uint8)*255
    boundary_as_raster_invert = rasterio.features.geometry_mask([boundary], (ncol,nrow), geoTransform, all_touched=0, invert=0).astype(np.uint8)*255
    
    #find the boundary
    boundary_as_raster_boundary = binary_dilation(boundary_as_raster_boundary,np.ones((3,3))).astype(np.uint8)*255
    boundary_as_raster_invert = binary_dilation(boundary_as_raster_invert,np.ones((3,3))).astype(np.uint8)*255
    boundary_line = np.bitwise_and(boundary_as_raster_boundary,boundary_as_raster_invert)
    
    
    boundary_idxs = np.where(boundary_line>0)
    
    field = cv2.merge((band_B,band_G,band_R))
    
    field [:,:,0][boundary_idxs] = 0
    field [:,:,1][boundary_idxs] = 0
    field [:,:,2][boundary_idxs] = 255
    
    if save_path != None:
        cv2.imwrite(save_path,field)
   
    return field 
    
    
def example(num = 129) :
    
    output = '../example_output'
    shp_path = '../fields/fields.shp'
    rst_path = '../rasters/%d.tif' % num
    
    
    data, epsg, field_definition = read_shpfile(shp_path)
    
    
    # find the boundary which corresponds to the raster
    target_boundary = None
    for d in data:
        if d[9] == num:
            target_boundary = json.loads(d[0])
            break
    
    
    draw_boundary(rst_path,target_boundary,save_path =os.path.join(output,'draw_boundary%d.jpg'%num) )
    extract_field(rst_path,target_boundary,save_path =os.path.join(output,'extract_field%d.jpg'%num) )

#example()