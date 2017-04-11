# -*- coding: utf-8 -*-
"""
Granger causality.

The purpose of this script is to calculate the 'Granger causality' difference
score between the average time course within an ROI and all other voxels in an
nii file.

See <http://nipy.org/nitime/examples/granger_fmri.html> for more information.

(C) Ingo Marquardt, 11.04.2017
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

from pipeline import pipeline

# *****************************************************************************
# *** Define parameters

# Subject IDs (as in file path):
lstSubId = ['S01', 'S02']

# Condition IDs:
lstConId = ['Left_FEF_Session', 'Right_FEF_Session']

# ROI IDs:
lstRoiId = ['Left_FEF', 'Right_FEF']

# Run IDs:
lstRunId = ['1', '2', '3']

# Base path for nii files (with subject ID, condition ID, and run number left
# open):
strPathNii = '/home/john/Desktop/neurofeedback_data/{}/{}/NF_Run_{}/filtered_func_data.nii.gz'  #noqa

# Path to ROI mask (subject ID, condition ID, and ROI ID left open):)
strPathRoi = '/home/john/Desktop/neurofeedback_data/{}/{}/ROIs/{}/conjunction.nii.gz'  #noqa

# Path to brain mask for functional data (Granger causality only calculated
# within mask):
# strPathMsk = ''  #noqa

# Output directory (subject ID and condition ID left open):
strPathOut = '/home/john/Desktop/neurofeedback_data/{}/{}/'  #noqa

# Parameters relavant for Granger causality analysis:

# Volume TR of input data [s]:
varTr = 2.0

# Bounds on the frequency band of interest [sˆ-1]:
varFreqMin = 0.02
varFreqMax = 0.15

# Intensity cutoff value for functional data (voxels with temporal mean below
# the threshold will be ignored):
varIntCtf = 100.0

# Title for plot of mean ROI time course:
strTitle = 'ROI time course'

# Limits of y-axis:
varYmin = -3.0
varYmax = 3.0

# Label for axes:
strXlabel = 'Time [volumes]'
strYlabel = 'Percent signal change'

# Figure scaling factor:
varDpi = 96.0

# Parallelisation factor (i.e. number of threads to create):
varPar = 10
# *****************************************************************************


# *****************************************************************************
# *** Run Granger analysis

print('-Granger analysis')

# Loop through subjects:
for strSubId in lstSubId:

    print('--Subject: ' + strSubId)

    # Loop through conditions:
    for strConId in lstConId:

        print('---Condition: ' + strConId)

        # Loop through reference ROIs:
        for strRoiId in lstRoiId:

            print('----Reference ROI: ' + strRoiId)

            # Create list with paths of all runs of current session:
            lstPathNii = [strPathNii.format(strSubId,
                                            strConId,
                                            strRunId) for strRunId in lstRunId]

            # Path of reference ROI:
            strPathRoiTmp = strPathRoi.format(strSubId, strConId, strRoiId)

            # Output path:
            strPathOutTmp = strPathOut.format(strSubId, strConId)

            # Session ID (used for output file names, subject ID, condition ID,
            # and ROI ID left open):
            strSessionId = '{}_{}_{}'.format(strSubId, strConId, strRoiId)

            # Call pipeline function and perform analysis:
            pipeline(lstPathNii,
                     strPathRoiTmp,
                     strPathOutTmp,
                     strSessionId,
                     varTr,
                     varFreqMin,
                     varFreqMax,
                     varPar=varPar,
                     varIntCtf=varIntCtf,
                     varDpi=varDpi,
                     varYmin=varYmin,
                     varYmax=varYmax,
                     strXlabel=strXlabel,
                     strYlabel=strYlabel,
                     strTitle=strTitle)
# *****************************************************************************
