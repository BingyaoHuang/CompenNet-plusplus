

[CompenNet++: End-to-end Full Projector Compensation][3] Dataset
===

This folder should contain CompenNet++ dataset. You can download and extract our [dataset (~10G)][5] here. 


## Folder Structure

    ├─init                     # CompenNet initialization image, i.e., \dot{s} in Eq 13.
    ├─light1                   # lighting level
    │  ├─pos1                  # cam, prj, surface pose
    │  │  ├─cloud_np           # surface texture, _np stands for nonplanar
    │  │  │  ├─cam             # camera-captured images or in images camera's view
    │  │  │  │  ├─desire       # desired effect, i.e., Fig. 1 (f)
    │  │  │  │  │  └─test      # testing images affine warped to displayable area, i.e., z' Fig. 5.
    │  │  │  │  ├─raw          # camera-captured raw images (w/o warping)
    │  │  │  │  │  ├─ref       # plain color images, ref/img_0126.png is \tilde(s)
    │  │  │  │  │  ├─sl        # SL images to convert raw->warpSL (two-step methods)
    │  │  │  │  │  ├─test      # CompenNet++ validation images \tilde(y)
    │  │  │  │  │  └─train     # CompenNet++ training images, \tilde(x)
    │  │  │  │  └─warpSL       # SL warped images, only used by two-step methods
    │  │  │  │      ├─ref      # SL warped raw/ref
    │  │  │  │      ├─test     # SL warped raw/test
    │  │  │  │      └─train    # SL warped raw/train
    │  │  │  └─prj             # compensation images (z*) of cam/desire/test (z') are saved here
    │  .  └─lavender_np        # another setup with another surface texture
    │  .  .
    │  .  .
    ├─light2                   # another lighting level
    .  .
    .  .
    .  .
    ├─ref                      # projector input plain color images, ref/img_gray.png is x0
    ├─sl                       # projector input SL images (see captured in cam/raw/sl) 
    ├─test                     # projector input validation images, i.e., y
    └─train                    # projector input training images, i.e., x

## Citation
    @inproceedings{huang2019compennet++,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {CompenNet++: End-to-end Full Projector Compensation},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019} }

    @inproceedings{huang2019compennet,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {End-To-End Projector Photometric Compensation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019} }

## Acknowledgments
We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
We thank the authors of the colorful textured sampling images. 

[1]: https://www3.cs.stonybrook.edu/~hling/publication/CompenNet++.pdf
[2]: https://www3.cs.stonybrook.edu/~hling/publication/CompenNet++_sup-high-res.pdf
[3]: https://github.com/BingyaoHuang/CompenNet-plusplus
[4]: https://github.com/BingyaoHuang/CompenNet
[5]: http://bit.ly/CompenNet-plusplus_Dataset