# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rajat/Desktop/Summer-Internship/week3/shot

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rajat/Desktop/Summer-Internship/week3/shot/build

# Include any dependencies generated for this target.
include CMakeFiles/shot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/shot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/shot.dir/flags.make

CMakeFiles/shot.dir/shot.cpp.o: CMakeFiles/shot.dir/flags.make
CMakeFiles/shot.dir/shot.cpp.o: ../shot.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rajat/Desktop/Summer-Internship/week3/shot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/shot.dir/shot.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/shot.dir/shot.cpp.o -c /home/rajat/Desktop/Summer-Internship/week3/shot/shot.cpp

CMakeFiles/shot.dir/shot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shot.dir/shot.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rajat/Desktop/Summer-Internship/week3/shot/shot.cpp > CMakeFiles/shot.dir/shot.cpp.i

CMakeFiles/shot.dir/shot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shot.dir/shot.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rajat/Desktop/Summer-Internship/week3/shot/shot.cpp -o CMakeFiles/shot.dir/shot.cpp.s

CMakeFiles/shot.dir/shot.cpp.o.requires:

.PHONY : CMakeFiles/shot.dir/shot.cpp.o.requires

CMakeFiles/shot.dir/shot.cpp.o.provides: CMakeFiles/shot.dir/shot.cpp.o.requires
	$(MAKE) -f CMakeFiles/shot.dir/build.make CMakeFiles/shot.dir/shot.cpp.o.provides.build
.PHONY : CMakeFiles/shot.dir/shot.cpp.o.provides

CMakeFiles/shot.dir/shot.cpp.o.provides.build: CMakeFiles/shot.dir/shot.cpp.o


# Object files for target shot
shot_OBJECTS = \
"CMakeFiles/shot.dir/shot.cpp.o"

# External object files for target shot
shot_EXTERNAL_OBJECTS =

shot: CMakeFiles/shot.dir/shot.cpp.o
shot: CMakeFiles/shot.dir/build.make
shot: /usr/lib/x86_64-linux-gnu/libboost_system.so
shot: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
shot: /usr/lib/x86_64-linux-gnu/libboost_thread.so
shot: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
shot: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
shot: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
shot: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
shot: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
shot: /usr/lib/x86_64-linux-gnu/libboost_regex.so
shot: /usr/lib/x86_64-linux-gnu/libpthread.so
shot: /usr/local/lib/libpcl_common.so
shot: /usr/local/lib/libpcl_octree.so
shot: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
shot: /usr/local/lib/libpcl_kdtree.so
shot: /usr/local/lib/libpcl_search.so
shot: /usr/local/lib/libpcl_sample_consensus.so
shot: /usr/local/lib/libpcl_filters.so
shot: /usr/local/lib/libpcl_io.so
shot: /usr/local/lib/libpcl_features.so
shot: /usr/local/lib/libpcl_registration.so
shot: /usr/local/lib/libpcl_ml.so
shot: /usr/local/lib/libpcl_segmentation.so
shot: /usr/local/lib/libpcl_keypoints.so
shot: /usr/lib/x86_64-linux-gnu/libqhull.so
shot: /usr/local/lib/libpcl_surface.so
shot: /usr/local/lib/libpcl_recognition.so
shot: /usr/local/lib/libpcl_visualization.so
shot: /usr/local/lib/libpcl_outofcore.so
shot: /usr/local/lib/libpcl_stereo.so
shot: /usr/local/lib/libpcl_people.so
shot: /usr/local/lib/libpcl_tracking.so
shot: /usr/lib/x86_64-linux-gnu/libboost_system.so
shot: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
shot: /usr/lib/x86_64-linux-gnu/libboost_thread.so
shot: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
shot: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
shot: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
shot: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
shot: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
shot: /usr/lib/x86_64-linux-gnu/libboost_regex.so
shot: /usr/lib/x86_64-linux-gnu/libpthread.so
shot: /usr/lib/x86_64-linux-gnu/libqhull.so
shot: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
shot: /usr/local/lib/libvtkFiltersSMP-7.1.so.1
shot: /usr/local/lib/libvtkFiltersTexture-7.1.so.1
shot: /usr/local/lib/libvtkFiltersHyperTree-7.1.so.1
shot: /usr/local/lib/libvtkIOTecplotTable-7.1.so.1
shot: /usr/local/lib/libvtkImagingStatistics-7.1.so.1
shot: /usr/local/lib/libvtkIOEnSight-7.1.so.1
shot: /usr/local/lib/libvtkFiltersSelection-7.1.so.1
shot: /usr/local/lib/libvtkFiltersProgrammable-7.1.so.1
shot: /usr/local/lib/libvtkIOExport-7.1.so.1
shot: /usr/local/lib/libvtkIOParallelXML-7.1.so.1
shot: /usr/local/lib/libvtkIOAMR-7.1.so.1
shot: /usr/local/lib/libvtkFiltersAMR-7.1.so.1
shot: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
shot: /usr/local/lib/libvtkImagingMorphological-7.1.so.1
shot: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.1.so.1
shot: /usr/local/lib/libvtkDomainsChemistry-7.1.so.1
shot: /usr/local/lib/libvtkIOPLY-7.1.so.1
shot: /usr/local/lib/libvtkIOSQL-7.1.so.1
shot: /usr/local/lib/libvtkImagingStencil-7.1.so.1
shot: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
shot: /usr/local/lib/libvtkIOImport-7.1.so.1
shot: /usr/local/lib/libvtkViewsInfovis-7.1.so.1
shot: /usr/local/lib/libvtkChartsCore-7.1.so.1
shot: /usr/local/lib/libvtkRenderingLabel-7.1.so.1
shot: /usr/local/lib/libvtkFiltersGeneric-7.1.so.1
shot: /usr/local/lib/libvtkIOVideo-7.1.so.1
shot: /usr/local/lib/libvtkIOInfovis-7.1.so.1
shot: /usr/local/lib/libvtklibxml2-7.1.so.1
shot: /usr/local/lib/libvtkFiltersVerdict-7.1.so.1
shot: /usr/local/lib/libvtkIOMovie-7.1.so.1
shot: /usr/local/lib/libvtkIOLSDyna-7.1.so.1
shot: /usr/local/lib/libvtkFiltersPoints-7.1.so.1
shot: /usr/local/lib/libvtkRenderingImage-7.1.so.1
shot: /usr/local/lib/libvtkGeovisCore-7.1.so.1
shot: /usr/local/lib/libvtkInteractionImage-7.1.so.1
shot: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.1.so.1
shot: /usr/local/lib/libvtkIOExodus-7.1.so.1
shot: /usr/local/lib/libvtkIOMINC-7.1.so.1
shot: /usr/local/lib/libvtkIOParallel-7.1.so.1
shot: /usr/local/lib/libvtkFiltersFlowPaths-7.1.so.1
shot: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
shot: /usr/local/lib/libvtkFiltersParallelImaging-7.1.so.1
shot: /usr/local/lib/libpcl_common.so
shot: /usr/local/lib/libpcl_octree.so
shot: /usr/local/lib/libpcl_kdtree.so
shot: /usr/local/lib/libpcl_search.so
shot: /usr/local/lib/libpcl_sample_consensus.so
shot: /usr/local/lib/libpcl_filters.so
shot: /usr/local/lib/libpcl_io.so
shot: /usr/local/lib/libpcl_features.so
shot: /usr/local/lib/libpcl_registration.so
shot: /usr/local/lib/libpcl_ml.so
shot: /usr/local/lib/libpcl_segmentation.so
shot: /usr/local/lib/libpcl_keypoints.so
shot: /usr/local/lib/libpcl_surface.so
shot: /usr/local/lib/libpcl_recognition.so
shot: /usr/local/lib/libpcl_visualization.so
shot: /usr/local/lib/libpcl_outofcore.so
shot: /usr/local/lib/libpcl_stereo.so
shot: /usr/local/lib/libpcl_people.so
shot: /usr/local/lib/libpcl_tracking.so
shot: /usr/local/lib/libvtkRenderingGL2PSOpenGL2-7.1.so.1
shot: /usr/local/lib/libvtkgl2ps-7.1.so.1
shot: /usr/local/lib/libvtksqlite-7.1.so.1
shot: /usr/local/lib/libvtkverdict-7.1.so.1
shot: /usr/local/lib/libvtkoggtheora-7.1.so.1
shot: /usr/local/lib/libvtkViewsCore-7.1.so.1
shot: /usr/local/lib/libvtkInfovisLayout-7.1.so.1
shot: /usr/local/lib/libvtkInfovisCore-7.1.so.1
shot: /usr/local/lib/libvtkproj4-7.1.so.1
shot: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
shot: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
shot: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
shot: /usr/local/lib/libvtkImagingColor-7.1.so.1
shot: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
shot: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
shot: /usr/local/lib/libvtkIOXML-7.1.so.1
shot: /usr/local/lib/libvtkImagingMath-7.1.so.1
shot: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
shot: /usr/local/lib/libvtkexpat-7.1.so.1
shot: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
shot: /usr/local/lib/libvtkIOGeometry-7.1.so.1
shot: /usr/local/lib/libvtkjsoncpp-7.1.so.1
shot: /usr/local/lib/libvtkIONetCDF-7.1.so.1
shot: /usr/local/lib/libvtkexoIIc-7.1.so.1
shot: /usr/local/lib/libvtkNetCDF_cxx-7.1.so.1
shot: /usr/local/lib/libvtkNetCDF-7.1.so.1
shot: /usr/local/lib/libvtkhdf5_hl-7.1.so.1
shot: /usr/local/lib/libvtkhdf5-7.1.so.1
shot: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
shot: /usr/local/lib/libvtkIOImage-7.1.so.1
shot: /usr/local/lib/libvtkpng-7.1.so.1
shot: /usr/local/lib/libvtkDICOMParser-7.1.so.1
shot: /usr/local/lib/libvtkmetaio-7.1.so.1
shot: /usr/local/lib/libvtktiff-7.1.so.1
shot: /usr/local/lib/libvtkjpeg-7.1.so.1
shot: /usr/lib/x86_64-linux-gnu/libm.so
shot: /usr/lib/x86_64-linux-gnu/libSM.so
shot: /usr/lib/x86_64-linux-gnu/libICE.so
shot: /usr/lib/x86_64-linux-gnu/libX11.so
shot: /usr/lib/x86_64-linux-gnu/libXext.so
shot: /usr/lib/x86_64-linux-gnu/libXt.so
shot: /usr/local/lib/libvtkglew-7.1.so.1
shot: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
shot: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
shot: /usr/local/lib/libvtkfreetype-7.1.so.1
shot: /usr/local/lib/libvtkFiltersParallel-7.1.so.1
shot: /usr/local/lib/libvtkRenderingCore-7.1.so.1
shot: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
shot: /usr/local/lib/libvtkCommonColor-7.1.so.1
shot: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
shot: /usr/local/lib/libvtkFiltersSources-7.1.so.1
shot: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
shot: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
shot: /usr/local/lib/libvtkFiltersCore-7.1.so.1
shot: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
shot: /usr/local/lib/libvtkParallelCore-7.1.so.1
shot: /usr/local/lib/libvtkIOLegacy-7.1.so.1
shot: /usr/local/lib/libvtkIOCore-7.1.so.1
shot: /usr/local/lib/libvtkzlib-7.1.so.1
shot: /usr/local/lib/libvtkFiltersImaging-7.1.so.1
shot: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
shot: /usr/local/lib/libvtkImagingSources-7.1.so.1
shot: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
shot: /usr/local/lib/libvtkImagingFourier-7.1.so.1
shot: /usr/local/lib/libvtkImagingCore-7.1.so.1
shot: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
shot: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
shot: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
shot: /usr/local/lib/libvtkCommonMisc-7.1.so.1
shot: /usr/local/lib/libvtkCommonMath-7.1.so.1
shot: /usr/local/lib/libvtkCommonSystem-7.1.so.1
shot: /usr/local/lib/libvtkCommonCore-7.1.so.1
shot: /usr/local/lib/libvtksys-7.1.so.1
shot: /usr/local/lib/libvtkalglib-7.1.so.1
shot: CMakeFiles/shot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rajat/Desktop/Summer-Internship/week3/shot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable shot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/shot.dir/build: shot

.PHONY : CMakeFiles/shot.dir/build

CMakeFiles/shot.dir/requires: CMakeFiles/shot.dir/shot.cpp.o.requires

.PHONY : CMakeFiles/shot.dir/requires

CMakeFiles/shot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/shot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/shot.dir/clean

CMakeFiles/shot.dir/depend:
	cd /home/rajat/Desktop/Summer-Internship/week3/shot/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rajat/Desktop/Summer-Internship/week3/shot /home/rajat/Desktop/Summer-Internship/week3/shot /home/rajat/Desktop/Summer-Internship/week3/shot/build /home/rajat/Desktop/Summer-Internship/week3/shot/build /home/rajat/Desktop/Summer-Internship/week3/shot/build/CMakeFiles/shot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/shot.dir/depend

