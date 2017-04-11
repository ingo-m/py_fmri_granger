# -*- coding: utf-8 -*-
"""
Granger analysis.

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
import nitime.analysis as nta
import nitime.timeseries as ts


def granger(idxPrc,  #noqa
            vecRoiMeanPsc,
            aryFuncChnk,
            varTr,
            varFreqMin,
            varFreqMax,
            varPar,
            queOut):
    """
    Calculate difference ('directionality') of 'Granger causality'.

    Parameters
    ----------
    idxPrc : int
        Process ID. This function can be used with the multiprocessing module.
        In order to organise the outputs, a process ID is attached to the
        outputs of this function.
    vecRoiMeanPsc : np.array
        Mean time course within reference region of interest (ROI). 'Granger
        causality' will be calculate with respect to this reference time
        course.
    aryFuncChnk : np.array
        Voxel time courses for which 'Granger causality' will be calculated, of
        the form: aryFuncChnk[voxelID, volume]
    varTr : float
        Volume TR of nii data.
    varFreqMin : float
        Lower bound on the frequency band of interest [s^-1].
    varFreqMax : float
        Upper bound on the frequency band of interest [s^-1].
    varPar : int
        Number of parallel processes that are run in parallel (only used for
        status update here).
    queOut : multiprocessing.Queue
        Queue object of the multiprocessing object. Results are put into the
        queue.

    Returns
    -------
    lstOut : list
        Output list containing process ID (`int`) and one-dimensional array
        with difference in 'Granger causality' for all voxels (1D `np.array`).

    Notes
    -----
    The difference in 'Granger causality' is the 'Granger causality' from x to
    y minus the 'Granger causality' from y to x.
    """
    # Number of voxels in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # List of 'analysis pairs'. Since the Granger analysis is done separately
    # for each voxel, we only have one 'pair'.
    lstPairs = [(0, 1)]

    # List for results, i.e. for Granger difference score ((X->Y) - (Y->X)),
    # per voxel:
    vecGrgDiff = np.zeros(varNumVoxChnk)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming Granger analysis
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Total number of voxels (all chunks in all processes together):
        varNumVoxAllPrc = int(varNumVoxChnk * varPar)

        # Vector with voxel indicies at which to give status feedback:
        vecStatVox = np.linspace(0,
                                 varNumVoxChnk,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatVox = np.ceil(vecStatVox)
        vecStatVox = vecStatVox.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through voxels (Granger analysis throws an error if a large number
    # of time courses is passed to it):
    for idxVox in range(0, varNumVoxChnk):

        # Status indicator (only used in the first of the parallel processes):
        if idxPrc == 0:

            # Status indicator:
            if varCntSts02 == vecStatVox[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatVox[varCntSts01] * varPar) +
                             ' voxels out of ' +
                             str(varNumVoxAllPrc))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # Put the reference ROI time course and the current voxel time courses
        # into one array (necessary for creation of the nitime 'timeseries'
        # object).
        aryTmp = np.vstack((vecRoiMeanPsc, aryFuncChnk[idxVox, :]))

        # Initialize TimeSeries object:
        objTimeSeries = ts.TimeSeries(aryTmp,
                                      sampling_interval=varTr,
                                      time_unit='s')

        # We initialize the GrangerAnalyzer object, while specifying the order
        # of the autoregressive model to be 1 (predict the current behavior of
        # the time-series based on one time-point back).
        objGrg = nta.GrangerAnalyzer(objTimeSeries,
                                     ij=lstPairs,
                                     order=1)

        # We are only interested in the physiologically relevant frequency band
        # (approximately 0.02 to 0.15 Hz):
        vecFreq = np.where((objGrg.frequencies > varFreqMin) *
                           (objGrg.frequencies < varFreqMax))[0]

        # The difference in Granger causality (the result is a 2-by-2 array.
        # All entries but one are 'nan' because we chose to only make one
        # comparison (between the reference average ROI time course, and the
        # current voxel time course).
        vecGrgDiff[idxVox] = \
            np.mean(np.subtract(objGrg.causality_xy[:, :, vecFreq],
                                objGrg.causality_yx[:, :, vecFreq]),
                    -1)[1, 0]

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc, vecGrgDiff]

    queOut.put(lstOut)
