#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:01:41 2023

@author: tauro
"""

def regrid2d(xcor_in, ycor_in, data_in, variance_in, quality_in, weights_in, xcor_out, ycor_out,
                divide_by_pixel_area=True, dorner_variance_scaling=True,
                print_progress=False, logger=None, logging_level='info'):

    import Polygon3 as Polygon
    import numpy
    """ Method to project data from a continuous input grid of 2D pixels on a continuous output 2D grid of pixels.
​
    This function loops through each pixel of the input and output grids, determines the fraction of the area of
    the input pixel covered by the output pixel (if any) and uses this value to assign a fraction of the value of
    the input pixel to the output pixel. It uses the Polygon package to compute the area of the overlap between a
    pair of input and output pixels.
    The code is derived from nips.module.rect_module code written by Nora Luetzgendorf and follows the equations
    (4.2) from the PhD thesis of Bernard Dorner(2012).
​
    Parameters
    ----------
    xcor_in : numpy.ndarray
        2D float array containing the x-axis coordinates of the corners of the input pixel grid (must be continuous).
        Shape: (nx_in+1, ny_in+1) where (nx_in, ny_in) is the shape of the input data array.
    ycor_in : numpy.ndarray
        2D float array containing the y-axis coordinates of the corners of the input pixel grid (must be continuous).
        Shape: (nx_in+1, ny_in+1) where (nx_in, ny_in) is the shape of the input data array.
    data_in: numpy.ndarray or numpy.ma.array
        2D float array containing the input data. Data must be per unit of area.
        Shape: (nx_in, ny_in).
    variance_in: numpy.ndarray or numpy.ma.array
        2D array containing the input data variance information. Data must be per (unit of area)^2.
        Shape: (nx_in, ny_in).
    quality_in: numpy.ndarray or numpy.ma.array
        2D array containing the input data quality flags. Must be of type numpy.uint64.
        Shape: (nx_in, ny_in).
    weights_in: numpy.ndarray or numpy.ma.array
        2D array containing user-defined weights that will be applied to the input data and variance.
        Shape: (nx_in, ny_in).
    xcor_out : numpy.ndarray
        2D float array containing the x-axis coordinates of the corners of the output pixel grid (must be continuous).
        Shape: (nx_out+1, ny_out+1) where (nx_out, ny_out) is the shape of the ouput data array.
    ycor_out : numpy.ndarray
        2D float array containing the y-axis coordinates of the corners of the output pixel grid (must be continuous).
        Shape: (nx_out+1, ny_out+1) where (nx_out, ny_out) is the shape of the output data array.
    divide_by_pixel_area : bool
        if set to True, the data will be divided by the input pixel area and the variance by the square of the
        pixel area. Default: True.
    dorner_variance_scaling : bool
        if set to True, the fraction of the input pixel area will be used instead of its square when projecting the
        variance following equation 4.2 of Bernhard Dorner's PhD. Default: True.
    print_progress : bool
        if set to True, an ASCII progress bar will be printed on the standard output. Default: False.
​
    Returns
    -------
    A tuple (data_out, variance_out, quality_out, filling_factor_out, npix_out) of 5 numpy arrays of size
    (nx_out,ny_out).
    data_out : output data array. Note that if divide_by_pixel_area=True, the output data will be per units of
        pixel area.
    variance_out : output variance array. Note that the weighting scheme applied when projecting the variance
        depends on whether dorner_variance_scaling is True or False.
    quality_out : output quality array. It is the bitwise OR of the quality flags of all the input pixels
        contributing to the output pixel. The function does not perform any quality flagging for the projection
        itself (e.g. based on the the filling factor of the output pixel).
    filling_factor_out : output array containing the fraction of the output pixel area covered by the overlapping
        input pixels.
    npix_out : output array indicating how many (valid = unmasked) input pixels overlapped with the output pixel.
​
    """
    # =============================================================================================================
    # Paranoid checks
    # =============================================================================================================

    shape_in = numpy.shape(data_in)
    nx_in = shape_in[0]
    ny_in = shape_in[1]
    ma_data_in = numpy.ma.array(data_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    ma_variance_in = numpy.ma.array(variance_in).harden_mask()
    # ------------------------------------------------------------------------------------------------------------
    ma_quality_in = numpy.ma.array(quality_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    ma_weights_in = numpy.ma.array(weights_in).harden_mask()
    # -------------------------------------------------------------------------------------------------------------
    work = numpy.shape(xcor_out)
    if (xcor_out[-1,0] >= xcor_out[0,0]):
        xdir_out = 1
    else:
        xdir_out = -1
    nx_out = work[0] - 1
    ny_out = work[1] - 1
    shape_out = (nx_out, ny_out)
    # -------------------------------------------------------------------------------------------------------------
    if (ycor_out[0,-1] >= ycor_out[0,0]):
        ydir_out = 1
    else:
        ydir_out = -1

    # =============================================================================================================
    # Extracting some information from the xcor_out and ycor_out arrays
    # =============================================================================================================
    xmax_out = xcor_out.max()
    ymax_out = ycor_out.max()
    xmin_out = xcor_out.min()
    ymin_out = ycor_out.min()
    envelope_out = Polygon.Polygon(((xmin_out, ymin_out), (xmax_out, ymin_out),
                                    (xmax_out, ymax_out), (xmin_out, ymax_out)))
    # =============================================================================================================
    # Main loops
    # =============================================================================================================
    combined_mask_in = numpy.ma.getmaskarray(ma_data_in) | numpy.ma.getmaskarray(ma_variance_in) \
                    | numpy.ma.getmaskarray(ma_quality_in) | numpy.ma.getmaskarray(ma_weights_in)
    corners_pixel_in = numpy.zeros((4, 2))
    corners_pixel_out = numpy.zeros((4, 2))
    data_out = numpy.zeros(shape_out, dtype=ma_data_in.dtype)
    variance_out = numpy.zeros(shape_out, dtype=ma_variance_in.dtype)
    quality_out = numpy.zeros(shape_out, dtype=ma_quality_in.dtype)
    normfactor_out = numpy.zeros(shape_out, dtype=ma_data_in.dtype)
    normfactorsquare_out = numpy.zeros(shape_out, dtype=ma_data_in.dtype)
    filling_factor_out = numpy.zeros(shape_out, dtype=ma_data_in.dtype)
    npix_out = numpy.zeros(shape_out, dtype=int)
    ncount = 0
    # -------------------------------------------------------------------------------------------------------------
    # Loop on the input pixels - y-axis
    # -------------------------------------------------------------------------------------------------------------
    for iy_in in range(ny_in):
        if ((combined_mask_in[:, iy_in] == True).all()):
            continue
        # ---------------------------------------------------------------------------------------------------------
        # Loop on the input pixels - x-axis
        # ---------------------------------------------------------------------------------------------------------
        for ix_in in range(nx_in):
            # .....................................................................................................
            # Checking if the pixel is masked
            # .....................................................................................................
            if combined_mask_in[ix_in, iy_in]:
                continue
            # .....................................................................................................
            # Input weight
            # .....................................................................................................
            w_in = ma_weights_in[ix_in, iy_in]
            if w_in == 0:
                continue
            # .....................................................................................................
            # Initialising the polygon corresponding to the input pixel
            # .....................................................................................................
            corners_pixel_in[0, 0] = xcor_in[ix_in, iy_in]
            corners_pixel_in[1, 0] = xcor_in[ix_in + 1, iy_in]
            corners_pixel_in[2, 0] = xcor_in[ix_in + 1, iy_in + 1]
            corners_pixel_in[3, 0] = xcor_in[ix_in, iy_in + 1]
            corners_pixel_in[0, 1] = ycor_in[ix_in, iy_in]
            corners_pixel_in[1, 1] = ycor_in[ix_in + 1, iy_in]
            corners_pixel_in[2, 1] = ycor_in[ix_in + 1, iy_in + 1]
            corners_pixel_in[3, 1] = ycor_in[ix_in, iy_in + 1]
            polygon_pixel_in = Polygon.Polygon(corners_pixel_in)
            # Skip immediately if the input pixel is outside of the envelope of the output grid
            if ((envelope_out & polygon_pixel_in).area() == 0.0):
                continue
            area_pixel_in = polygon_pixel_in.area()
            # .....................................................................................................
            # Storing the input pixel values in work variables
            # .....................................................................................................
            if divide_by_pixel_area:
                d_in = data_in[ix_in, iy_in] / area_pixel_in
                v_in = variance_in[ix_in, iy_in] / area_pixel_in**2
            else:
                d_in = data_in[ix_in, iy_in]
                v_in = variance_in[ix_in, iy_in]
            q_in = quality_in[ix_in, iy_in]
            # .....................................................................................................
            # Isolating the output pixels potentially overlapping with the input one
            # .....................................................................................................
            bb_in = polygon_pixel_in.boundingBox()
            if (xdir_out == 1):
                if (ydir_out == 1):
                    valid_pixels_out = numpy.where((xcor_out[0:-1, 0:-1] <= bb_in[1]) &
                                                   (xcor_out[1:, 1:] >= bb_in[0]) &
                                                   (ycor_out[0:-1, 0:-1] <= bb_in[3]) &
                                                   (ycor_out[1:, 1:] >= bb_in[2]))
                else:
                    valid_pixels_out = numpy.where((xcor_out[0:-1, 0:-1] <= bb_in[1]) &
                                                   (xcor_out[1:, 1:] >= bb_in[0]) &
                                                   (ycor_out[0:-1, 0:-1] >= bb_in[2]) &
                                                   (ycor_out[1:, 1:] <= bb_in[3]))
            else:
                if (ydir_out == 1):
                    valid_pixels_out = numpy.where((xcor_out[0:-1, 0:-1] >= bb_in[0]) &
                                                   (xcor_out[1:, 1:] <= bb_in[1]) &
                                                   (ycor_out[0:-1, 0:-1] <= bb_in[3]) &
                                                   (ycor_out[1:, 1:] >= bb_in[2]))
                else:
                    valid_pixels_out = numpy.where((xcor_out[0:-1, 0:-1] >= bb_in[0]) &
                                                   (xcor_out[1:, 1:] <= bb_in[1]) &
                                                   (ycor_out[0:-1, 0:-1] >= bb_in[2]) &
                                                   (ycor_out[1:, 1:] <= bb_in[3]))

            # -----------------------------------------------------------------------------------------------------
            # Loop over the output pixels
            # -----------------------------------------------------------------------------------------------------
            for ix_out,iy_out in zip(valid_pixels_out[0], valid_pixels_out[1]):
                # .................................................................................................
                # Generating the polygon corresponding to the output pixel
                # .................................................................................................
                corners_pixel_out[0, 0] = xcor_out[ix_out, iy_out]
                corners_pixel_out[1, 0] = xcor_out[ix_out + 1, iy_out]
                corners_pixel_out[2, 0] = xcor_out[ix_out + 1, iy_out + 1]
                corners_pixel_out[3, 0] = xcor_out[ix_out, iy_out + 1]
                corners_pixel_out[0, 1] = ycor_out[ix_out, iy_out]
                corners_pixel_out[1, 1] = ycor_out[ix_out + 1, iy_out]
                corners_pixel_out[2, 1] = ycor_out[ix_out + 1, iy_out + 1]
                corners_pixel_out[3, 1] = ycor_out[ix_out, iy_out + 1]
                polygon_pixel_out = Polygon.Polygon(corners_pixel_out)
                area_pixel_out = polygon_pixel_out.area()
                # .................................................................................................
                # Projection
                # .................................................................................................
                oa = (polygon_pixel_in & polygon_pixel_out).area()
                if oa == 0.0:
                    continue
                filling_factor_out[ix_out, iy_out] = filling_factor_out[ix_out, iy_out] + oa / area_pixel_out
                npix_out[ix_out, iy_out] = npix_out[ix_out, iy_out] + 1
                fa = oa / area_pixel_in
                current_normfactor_out = normfactor_out[ix_out, iy_out] + fa * w_in
                data_out[ix_out, iy_out] = (d_in * fa * w_in + data_out[ix_out, iy_out]
                                            * normfactor_out[ix_out, iy_out]) / current_normfactor_out
                if (dorner_variance_scaling):
                    variance_out[ix_out, iy_out] = (v_in * fa * (w_in)**2 + variance_out[ix_out, iy_out]
                                            * normfactor_out[ix_out, iy_out]**2) / current_normfactor_out**2
   
                else:
                    variance_out[ix_out, iy_out] = (v_in * (fa * w_in)**2 + variance_out[ix_out, iy_out]
                                          * normfactor_out[ix_out, iy_out]**2) / current_normfactor_out**2
   
                quality_out[ix_out, iy_out] = quality_out[ix_out, iy_out] | q_in
                normfactor_out[ix_out, iy_out] = current_normfactor_out
    # =============================================================================================================
    # This is the end, my friend, the end
    # =============================================================================================================
    return (data_out, variance_out, quality_out, filling_factor_out, npix_out)








