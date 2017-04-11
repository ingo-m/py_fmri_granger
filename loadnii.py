# -*- coding: utf-8 -*-
"""
Load nii files.

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

import numpy as np  #noqa
import nibabel as nib


def load_nii(strPathIn):
    """
    Load nii files.

    Parameters
    ----------
    strPathIn : str
        Path of the nii file.

    Returns
    -------
    aryNii : np.array
        Three- or four-dimensional array with nii data.

    objHdr : nibabel header object
        Nibabel object with information from nifti header (such as image
        dimensions, resolution, etc.).

    aryAff : np.array
        Array with 'affine', i.e. information on image position.

    Notes
    -----
    This function uses nibabel to load the nii data. The nifti data put into a
    numpy array.
    """
    # print(('------Loading: ' + strPathIn))

    # Load nii file (this doesn't load the data into memory yet):
    objNii = nib.load(strPathIn)

    # Load data into array:
    aryNii = objNii.get_data()

    # Get headers:
    objHdr = objNii.header

    # Get 'affine':
    aryAff = objNii.affine

    # Output nii data as numpy array and header:
    return aryNii, objHdr, aryAff
