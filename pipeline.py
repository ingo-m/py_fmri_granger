# -*- coding: utf-8 -*-
"""
'Granger causality' analysis pipeline.

Function of py_fmri_granger library.
"""

# Part of the py_fmri_granger library.
# Copyright (C) 2017 Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import nibabel as nib
from loadnii import load_nii
import matplotlib.pyplot as plt
import multiprocessing as mp
from granger import granger


def pipeline(lstPathNii,
             strPathRoi,
             strPathOut,
             strSessionId,
             varTr,
             varFreqMin,
             varFreqMax,
             varPar=10,
             varIntCtf=100.0,
             varDpi=80.0,
             varYmin=-3.0,
             varYmax=3.0,
             strXlabel='Time [volumes]',
             strYlabel='Percent signal change',
             strTitle='Reference ROI time course',
             ):
    """
    Granger 'causality' analysis pipeline.

    Parameters
    ----------
    lstPathNii : list
        List with file paths of 4D nii files to be analysed.
    strPathRoi : str
        Path of reference ROI. The mean time course (mean across voxels) of
        this ROI is taken as the reference with respect to which 'Granger
        causality' is calculated in all other voxels.
    strPathOut : str
        Output path. Results will be saved here.
    strSessionId : str
        Output suffix (e.g. containing subject ID).
    varTr : float
        Volume TR of nii data.
    varFreqMin : float
        Lower bound on the frequency band of interest [s^-1].
    varFreqMax : float
        Upper bound on the frequency band of interest [s^-1].

    varPar : int
        Number of processes to run in parallel.
    varIntCtf : float
        Intensity cutoff value for functional data (voxels with temporal mean
        below the threshold will be ignored).
    varDpi : float
        Figure scaling factor.
    varYmin : float
        Minimum of y-axis for reference ROI time course plot.
    varYmax : float
        Maximum of y-axis for reference ROI time course plot.
    strXlabel : str
        Label for the x-axis.
    strYlabel : str
        Label for the y-axis.
    strTitle: str
        Figure title.

    Returns
    -------
    None : None
        This function has no return value.

    Notes
    -----
    (1) The reference ROI time course is plotted.
    (2) The difference 'Granger causality' between the reference time course
    and all voxels is computed. The difference in 'Granger causality' is the
    'Granger causality' from x to y minus the 'Granger causality' from y to x.
    (3) The result is saved in nii format.

    References
    ----------
    <http://nipy.org/nitime/examples/granger_fmri.html>
    """
    # *************************************************************************
    # *** Preparations
    print('---Preparations')

    print('------Loading nii data')

    # List for nii data of all runs:
    lstFunc = [None] * len(lstPathNii)

    # Loop through runs and load nii data:
    for idxRun in range(0, len(lstPathNii)):
        lstFunc[idxRun], hdrFunc, aryAffFunc = load_nii(lstPathNii[idxRun])

    # Concatenate all runs along time dimension (assumed to be fourth
    # dimension, i.e. indexed as 3rd axis):
    aryFunc = np.concatenate(lstFunc, axis=3)
    # Delete original list of data (the concatenation created a hard copy):
    del(lstFunc)

    # Load the ROI nii file:
    aryRoi, hdrRoi, aryAffRoi = load_nii(strPathRoi)
    # aryMask, hdrMask, aryAffMask = load_nii(strPathMsk)

    print('------Normalising & taking mean ROI time course')

    # Take mean across time:
    aryFuncTmean = np.mean(aryFunc,
                           axis=3)

    # Save mean image (for quality control):
    niiTmean = nib.Nifti1Image(aryFuncTmean,
                               aryAffRoi,
                               header=hdrRoi)
    nib.save(niiTmean, (strPathOut + strSessionId + '_Tmean.nii.gz'))

    # Array with indicies of zeros in mean:
    aryLgcNotZero = np.not_equal(aryFuncTmean, 0.0)

    # Normalise the data to percent signal change (with saveguard against
    # division by zero, giving zero for such instances):
    aryFunc = np.subtract(np.multiply(np.divide(aryFunc,
                                                aryFuncTmean[:, :, :, None],
                                                out=np.zeros(aryFunc.shape),
                                                where=aryLgcNotZero[:, :, :,
                                                                    None],
                                                ),
                                      100.0),
                          100.0)

    # Set voxels with zeros value in mean image to zero in PSC array:
    aryFunc[np.logical_not(aryLgcNotZero), :] = 0.0

    print('------Extract ROI time course')

    # Create average time course within ROI. Start by creating a logical array
    # with indicies of non-zero voxels in mask:
    aryRoi = np.greater(aryRoi, 0.0)

    # Take mean across spatial dimension, resulting in the mean time course:
    vecRoiMeanPsc = np.mean(aryFunc[aryRoi, :],
                            axis=0)

    # *************************************************************************
    # *** Plot ROI time course

    print('---Creating ROI time course plot')

    # Create figure:
    fgr01 = plt.figure(figsize=(800.0/varDpi, 500.0/varDpi),
                       dpi=varDpi)
    # Create axis:
    axs01 = fgr01.add_subplot(111)

    # Vector for x-data:
    vecX = range(0, vecRoiMeanPsc.shape[0])

    # Plot depth profile for current input file:
    pltTmp = axs01.plot(vecX,  #noqa
                        vecRoiMeanPsc,
                        color=[0.2, 0.5, 0.7],
                        alpha=0.8,
                        linewidth=1.5,
                        antialiased=True)

    # Set y-axis range:
    axs01.set_ylim([varYmin, varYmax])

    # Adjust labels & title for axis 1:
    axs01.tick_params(labelsize=11)
    axs01.set_xlabel(strXlabel,
                     fontsize=12)
    axs01.set_ylabel(strYlabel,
                     fontsize=12)
    axs01.set_title(strTitle,
                    fontsize=12)

    # # Add vertical grid lines:
    #    axs01.xaxis.grid(which=u'major',
    #                     color=([0.5,0.5,0.5]),
    #                     linestyle='-',
    #                     linewidth=0.2)

    # Save figure:
    fgr01.savefig((strPathOut + strSessionId + '_ROI_time_course.png'),
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  transparent=False,
                  frameon=None)

    # *************************************************************************
    # *** Granger causality analysis

    print('---Granger causality analysis')

    print('------Preparing parallelisation')

    # Empty list for results (Granger causality difference):
    lstRes = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Counter for parallel processes:
    # varCntPar = 0

    # Counter for output of parallel processes:
    # varCntOut = 0

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Get shape of functional nii data:
    vecNiiShp = aryFunc.shape

    # Total number of voxels:
    varNumVoxTlt = (vecNiiShp[0] * vecNiiShp[1] * vecNiiShp[2])

    # Reshape functional nii data:
    aryFunc = np.reshape(aryFunc, [varNumVoxTlt, vecNiiShp[3]])

    # Reshape mask:
    # aryMask = np.reshape(aryMask, varNumVoxTlt)

    # Reshape ROI:
    aryRoi = np.reshape(aryRoi, varNumVoxTlt)

    # Reshape mean functional image:
    aryFuncTmean = np.reshape(aryFuncTmean, varNumVoxTlt)

    # Logical test for voxel inclusion: is the voxel value greater than zero in
    # the mask, is the mean of the functional time series above the cutoff
    # value, and is the voxel outside the reference ROI?
    # aryLgc = np.multiply(np.greater(aryMask, 0),
    #                      np.greater(aryFuncTmean, varIntCtf))
    # aryLgc = np.multiply(aryLgc,
    #                      np.logical_not(aryRoi))
    aryLgc = np.multiply(np.greater(aryFuncTmean, varIntCtf),
                         np.logical_not(aryRoi))

    # Array with functional data for which conditions (mask inclusion and
    # cutoff value) are fullfilled:
    aryFunc = aryFunc[aryLgc, :]

    # Number of voxels for which Granger analysis will be performed:
    varNumVoxInc = aryFunc.shape[0]

    print('------Number of voxels on which ranger analysis will be performed: '
          + str(varNumVoxInc))

    # List into which the chunks of functional data for the parallel processes
    # will be put:
    lstFunc = [None] * varPar

    # Vector with the indicies at which the functional data will be separated
    # in order to be chunked up for the parallel processes:
    vecIdxChnks = np.linspace(0,
                              varNumVoxInc,
                              num=varPar,
                              endpoint=False)
    vecIdxChnks = np.hstack((vecIdxChnks, varNumVoxInc))

    # Put functional data into chunks:
    for idxChnk in range(0, varPar):
        # Index of first voxel to be included in current chunk:
        varTmpChnkSrt = int(vecIdxChnks[idxChnk])
        # Index of last voxel to be included in current chunk:
        varTmpChnkEnd = int(vecIdxChnks[(idxChnk+1)])
        # Put voxel array into list:
        lstFunc[idxChnk] = aryFunc[varTmpChnkSrt:varTmpChnkEnd, :]

    # We don't need the original array with the functional data anymore:
    del(aryFunc)

    print('------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=granger,
                                     args=(idxPrc,
                                           vecRoiMeanPsc,
                                           lstFunc[idxPrc],
                                           varTr,
                                           varFreqMin,
                                           varFreqMax,
                                           varPar,
                                           queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstRes[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('------Prepare results for export')

    # Create list for vectors with analysis results, in order to put the
    # results into the correct order:
    lstGdiff = [None] * varPar

    # Put output into correct order:
    for idxRes in range(0, varPar):

        # Index of results (first item in output list):
        varTmpIdx = lstRes[idxRes][0]

        # Put fitting results into list, in correct order:
        lstGdiff[varTmpIdx] = lstRes[idxRes][1]

    # Concatenate output vectors (into the same order as the voxels that were
    # included in the analysis):
    vecGdiff = np.zeros(0)
    for idxRes in range(0, varPar):
        vecGdiff = np.append(vecGdiff, lstGdiff[idxRes])

    # Delete unneeded large objects:
    del(lstRes)
    del(lstGdiff)

    # Array for results that will be brought into the same shape as the
    # original nii data.
    aryGdiff = np.zeros((varNumVoxTlt, 1))

    # Put results form pRF finding into array (they originally needed to be
    # saved in a list due to parallelisation).
    aryGdiff[aryLgc, 0] = vecGdiff

    # Reshape pRF finding results:
    aryGdiff = np.reshape(aryGdiff,
                          [vecNiiShp[0],
                           vecNiiShp[1],
                           vecNiiShp[2],
                           1])

    # *************************************************************************
    # *** Save results

    # Create nii object for Granger difference score:
    niiGDiff = nib.Nifti1Image(aryGdiff,
                               aryAffRoi,
                               header=hdrRoi)
    # Save nii:
    nib.save(niiGDiff, (strPathOut + strSessionId + '_Gdiff.nii.gz'))
