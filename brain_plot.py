def plot_wholebrain(data, atlas, surface='white', cmap='viridis', vmin=None, vmax=None):
    '''
    Plot cortical surface and subcortical MNI volume

    Parameters
    ----------
    data : array of data (in same order as atlas)
    atlas : atlas name e.g 'brainnetome'
    surface : surface name e.g 'veryinflated'
    cmap : colormap
    vmin : minimum value for colormap
    vmax : maximum value for colormap
    show_edges : whether to show edges on the surface plot

    Returns
    -------
    fig : figure
    '''

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import nibabel as nib
    from neuromaps.datasets import fetch_fslr
    from brainspace.datasets import load_conte69
    from brainspace.mesh.mesh_io import read_surface
    from surfplot import Plot
    from nilearn import plotting, image
    from nilearn.image import smooth_img
    from PIL import Image
    
    # get the surface
    if surface == 'conte69':
        lh, rh = load_conte69()
    if surface == 'white':
        lh = read_surface('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/fs_LR.32k.L.white.surf.gii')
        rh = read_surface('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/fs_LR.32k.R.white.surf.gii')
    else:
        surfaces = fetch_fslr('32k')
        lh, rh = surfaces[surface]

    # if mean of data is < 1, multiply by 100
    # as nilearn plotting function doesn't like small numbers
    if np.nanmean(data) < 1:
        data = data * 100
        if vmin != None:
            vmin = vmin * 100
        if vmax != None:
            vmax = vmax * 100

    if vmin == None:
        vmin = np.min(data)
    if vmax == None:
        vmax = np.max(data)

    # make a dictionary out of the data array
    data_dict = {}
    for i in range(len(data)):
        data_dict[i+1] = data[i]

    

    if atlas == 'extended_schaefer_200':
        atlas_L = np.genfromtxt('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/schaefer_200_fslr32k_L.csv', delimiter=',')
        atlas_R = np.genfromtxt('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/schaefer_200_fslr32k_R.csv', delimiter=',')
        atlas_vol = nib.load('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/extended_schaefer_200.nii.gz')
        subcort_lim = 201 # 201-216 are subcortical regions in extended schaefer 200 atlas

    atlas_L_empty = atlas_L.copy()
    atlas_R_empty = atlas_R.copy()

    # use dictionary to change the values in the atlas to the data values
    for i in range(len(atlas_L)):
        if atlas_L[i] != 0:
            atlas_L[i] = data_dict[atlas_L[i]]
    for i in range(len(atlas_R)):
        if atlas_R[i] != 0:
            atlas_R[i] = data_dict[atlas_R[i]]

    # load MNI template from my file
    mni = nib.load('/rds/general/user/ab5621/home/Masters-Dissertation/Helper Files/mni_icbm152_t1_tal_nlin_asym_55_ext_brain.nii')
    mni = smooth_img(mni, fwhm=0.35) 

    # add subcortical region data to volume
    atlas_vol_data = atlas_vol.get_fdata()

    atlas_vol_data[atlas_vol_data < subcort_lim] = 0 # set any value < subcort_lim to 0

    non_zero_indices = np.nonzero(atlas_vol_data)
    for i, j, k in zip(*non_zero_indices):
        atlas_vol_data[i, j, k] = data_dict[atlas_vol_data[i, j, k]]

    # make into a nifti
    atlas_vol_nifti = nib.Nifti1Image(atlas_vol_data, atlas_vol.affine)

    # in the top left subplot, plot the surfplot for the left lateral view
    p = Plot(lh, views='lateral', brightness=.5)
    p.add_layer({'left': atlas_L}, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    # if show_sig != False:
    #     p.add_layer({'left': signodes_L}, cmap='binary_r', as_outline=True, cbar=False)
    fig1 = p.build()
    plt.close()

    # in the bottom left, plot the surfplot for the left medial view
    p = Plot(lh, views='medial', brightness=.5)
    p.add_layer({'left': atlas_L}, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    # if show_sig != False:
    #     p.add_layer({'left': signodes_L}, cmap='binary_r', as_outline=True, cbar=False)
    fig2 = p.build()
    plt.close()

    # in the top right, plot the surfplot for the right lateral view
    p = Plot(surf_rh=rh, views='lateral', brightness=.5)
    p.add_layer({'right': atlas_R}, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    # if show_sig != False:
    #     p.add_layer({'right': signodes_R}, cmap='binary_r', as_outline=True, cbar=False)
    fig3 = p.build()
    plt.close()

    # in the bottom right, plot the surfplot for the right medial view
    p = Plot(surf_rh=rh, views='medial', brightness=.5)
    p.add_layer({'right': atlas_R}, cmap=cmap, color_range=(vmin, vmax), cbar=False)
    # if show_sig != False:
    #     p.add_layer({'right': signodes_R}, cmap='binary_r', as_outline=True, cbar=False)
    fig4 = p.build()
    plt.close()

    # volplots

    # in the middle top plot, plot the MNI brain with subcortical regions at coordinates 9
    fig5 = plotting.plot_roi(atlas_vol_nifti, bg_img=mni,
                        display_mode='z', cut_coords=[7.4], alpha=.9, dim=0, resampling_interpolation='continuous',
                        cmap=cmap, annotate=False, vmin=vmin, vmax=vmax, black_bg=False)
    plt.close()

    # in the middle bottom plot, plot the MNI brain with subcortical regions at coordinates -3
    fig6 = plotting.plot_roi(atlas_vol_nifti, bg_img=mni,
                        display_mode='z', cut_coords=[-3], alpha=.9, dim=0, resampling_interpolation='continuous',
                        cmap=cmap, annotate=False, vmin=vmin, vmax=vmax, black_bg=False)
    plt.close()
    # instead of saving this to real files use temp
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()
    fig1.savefig(temp_dir.name + '/top_left.png', dpi=500)
    fig2.savefig(temp_dir.name + '/bottom_left.png', dpi=500)
    fig3.savefig(temp_dir.name + '/top_right.png', dpi=500)
    fig4.savefig(temp_dir.name + '/bottom_right.png', dpi=500)
    fig5.savefig(temp_dir.name + '/middle_top.png', dpi=500)
    fig6.savefig(temp_dir.name + '/middle_bottom.png', dpi=500)
    # read in the images
    img1 = Image.open(temp_dir.name + '/top_left.png')
    img2 = Image.open(temp_dir.name + '/bottom_left.png')
    img3 = Image.open(temp_dir.name + '/top_right.png')
    img4 = Image.open(temp_dir.name + '/bottom_right.png')
    img5 = Image.open(temp_dir.name + '/middle_top.png')
    img6 = Image.open(temp_dir.name + '/middle_bottom.png')


    # make a figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(img1)
    axes[1, 0].imshow(img2)
    axes[0, 2].imshow(img3)
    axes[1, 2].imshow(img4)
    axes[0, 1].imshow(img5)
    axes[1, 1].imshow(img6)
    for i in range(2):
        for j in range(3):
            axes[i, j].set_axis_off()
    axes[0, 0].set_xlim([450, 2630])
    axes[1, 0].set_xlim([450, 2630])
    axes[1, 2].set_xlim([450, 2630])
    axes[0, 2].set_xlim([450, 2630])
    axes[0, 0].set_ylim([2100, 400])
    axes[1, 0].set_ylim([2100, 400])
    axes[1, 2].set_ylim([2100, 400])
    axes[0, 2].set_ylim([2100, 400])
    axes[0, 1].set_xlim([-300,1390])
    axes[1, 1].set_xlim([-300,1390])

    # move the 3rd column closer to the middle
    fig.subplots_adjust(wspace=-0.2)
    fig.subplots_adjust(hspace=-0.2)

    for ax in axes[:, 0]:
        ax.set_zorder(1) # make the first column the top layer of the figure

    return fig



import numpy as np

# generate a random 232 length array between 0 and 1
data = np.random.rand(232)

# plot the whole brain with different surfaces and colormaps
fig_one = plot_wholebrain(data = data, atlas= 'extended_schaefer_200', cmap='Purples', vmin=0, vmax=0.0206)
fig_two = plot_wholebrain(data = data, atlas='extended_schaefer_200', cmap='inferno', vmin=0, vmax=0.0206)

fig_one.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Results/Plots/brain_plot_one.png', dpi=500)
fig_two.figure.savefig('/rds/general/user/ab5621/home/Masters-Dissertation/Results/Plots/brain_plot_two.png', dpi=500)



