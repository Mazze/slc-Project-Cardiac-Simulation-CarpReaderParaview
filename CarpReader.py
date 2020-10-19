from paraview.util.vtkAlgorithm import *

# TODO
#@smproxy.source(name="NCUBEImageOnTopographySource",
#       label="N-Cube Image On Topography Source")

# To add a reader, we can use the following decorators
#   @smproxy.source(name="PythonCSVReader", label="Python-based CSV Reader")
#   @smhint.xml("""<ReaderFactory extensions="csv" file_description="Numpy CSV files" />""")
# or directly use the "@reader" decorator.
# @smproperty.xml("""<OutputPort name="Header"     index="0" />""")
# @smproperty.xml("""<OutputPort name="Curves"     index="1" />""")
import numpy as np
import sys 

# IGB constances
FORMFEED = b'\x0c'
MAXLENGTH = 128*1024
# Save original file open
fopen = open
@smproxy.reader(name="CarpReader", label="IGB Reader",
                extensions="igb", file_description="IGB files")
class NCubeLASReader(VTKPythonAlgorithmBase):
    """A reader that reads a LAS Well Log file"""
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1)
        self._filename = None
        self._timesteps = None
        self._hdr_len = None
        self._hdr_content = None
        self._mesh= None

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="igb", file_description="Carb IGB files")
    def SetFileName(self, name):
        """Specify filename for the file to read."""
        if self._filename != name:
            self._filename = name
            self.Modified()

    # @smproperty.doublevector(name="Location", default_values=[0, 0, 0])
    # @smdomain.doublerange()
    # def SetLocation(self, x, y, z):
    #     self._x = x
    #     self._y = y
    #     self._z = z
    #     self.Modified()

    # @smproperty.doublevector(name="Azimuth", default_values=0)
    # @smdomain.doublerange(min=0, max=360)
    # def SetAzimuth(self, az):
    #     self._az = az
    #     self.Modified()
    def _openFile(self):
        return fopen(self._filename, 'rb')

    def _header_length(self):
        """
        Determine the length of the IGB file header, in bytes.

        Only relevant for read-mode files.

        Returns:
            (int) Number of bytes in the header
        """

        if self._hdr_len is not None:
            return self._hdr_len

        # Make sure at start
        fp=self._openFile()
        fp.seek(0, 0)

        # Get first byte
        count = 1
        byte = fp.read(1)

        while byte != FORMFEED:

            # Check EOF not reached
            if byte == '':
                raise Exception('File ended before header read')

            # Check we are not accumulating unreasonably large header
            if count > MAXLENGTH:
                raise Exception('Header exceeds {0} bytes'.format(MAXLENGTH))

            # Read next byte
            byte = fp.read(1)
            count += 1

        # Cache result
        self._hdr_len = count
        fp.close()
        return count

    def header(self):
        """
        Read the IGB file header and return as python dictionary.

        Returns:
            (dict) The contents of the file header
        """

        if self._hdr_content is not None:
            return self._hdr_content

        # Get header length
        hdr_len = self._header_length()
        fp=self._openFile()
        # Rewind file
        fp.seek(0, 0)
        hdr_str = fp.read(hdr_len - 1)
        fp.close()
        # Python 3 compatibility
        hdr_str = hdr_str.decode('utf-8')

        # Clean newline characters
        hdr_str = hdr_str.replace('\r', ' ')
        hdr_str = hdr_str.replace('\n', ' ')
        hdr_str = hdr_str.replace('\0', ' ')

        # Build dictionary of header content
        self._hdr_content = {}

        for part in hdr_str.split():

            key, value = part.split(':')

            if key in ['x', 'y', 'z', 't', 'bin', 'num', 'lut', 'comp']:
                self._hdr_content[key] = int(value)

            elif (key in ['facteur', 'zero', 'epais']
                    or key.startswith('org_')
                    or key.startswith('dim_')
                    or key.startswith('inc_')):
                self._hdr_content[key] = float(value)

            else:
                self._hdr_content[key] = value

        if 'inc_t' not in self._hdr_content:
            try:
                dim_t = self._hdr_content['dim_t']
                t = self._hdr_content['t']
            except KeyError:
                pass
            else:
                if t > 1:
                    self._hdr_content['inc_t'] = dim_t / (t - 1)

        return self._hdr_content
    def dtype(self):
        """
        Get a numpy-friendly data type for this file.

        Returns:
            (numpy.dtype) The numpy data type corresponding to the file contents
        """

        hdr = self.header()

        # Get numpy data type
        dtype = {'char':   np.byte,
                 'short':  np.short,
                 'int':    np.intc,
                 'long':   np.int_,
                 'ushort': np.ushort,
                 'uint':   np.uintc,
                 'float':  np.single,
                 'vec3f':  np.single,
                 'vec9f':  np.single,
                 'double': np.double}[hdr['type']]
        dtype = np.dtype(dtype)

        # Get python byte order string
        endian = {'little_endian': '<',
                  'big_endian':    '>'}[hdr['systeme']]

        # Return data type with correct order
        return dtype.newbyteorder(endian)

    def GetTimeFrame(self,t):
        """
        Return a numpy array of the file contents, for a given time

        The data is returned as a flat array. It is up to the user to use the
        header information to determine how to reshape the array, if desired.

        Returns:
            (numpy.ndarray) A numpy array with the file contents
        """
        header = self.header()
        # Sanity check
        assert self.header()['type'] in ['int','float', 'vec3f', 'vec9f'], \
               'Only int, float, vec3f and vec9f currently supported'

        # Move to start of content
        hdr_len = self._header_length()
        fp=self._openFile()
        offset = int(np.dtype(self.dtype()).itemsize*header.get('x')*t)
        print ("Offset %d"%offset)
        fp.seek(hdr_len+offset, 0)

        # if isinstance(self._fp, gzip.GzipFile):
        #     # Read remaining file
        #     byte_str = self._fp.read()
        #     # Create a numpy array view on content
        #     data = np.frombuffer(byte_str, dtype=self.dtype())

        # else:
            # Use more efficient direct read from file
            # This function uses the underlying C FILE pointer directly
        # print ("Np version ")
        # print (np.__version__)
        # print (self.dtype())
        # print ("Sinle size %d" % np.dtype(self.dtype()).itemsize )
        data = np.fromfile(fp, dtype=self.dtype(),count=header.get('x'))

        return data

    def _get_timesteps(self):
        print('_get_timesteps')
        header = self.header()
        self._timesteps = np.arange(header.get('t'))
        return self._timesteps.tolist()

    def _get_update_time(self, outInfo):
        print('_get_update_time')
        executive = self.GetExecutive()
        timesteps = self._get_timesteps()
        if timesteps is None or len(timesteps) == 0:
            print (" _get_update_time returns none  ")
            return None
        elif outInfo.Has(executive.UPDATE_TIME_STEP()) and len(timesteps) > 0:
            utime = outInfo.Get(executive.UPDATE_TIME_STEP())
            dtime = timesteps[0]
            for atime in timesteps:
                if atime > utime:
                    return dtime
                else:
                    dtime = atime
            return dtime
        else:
            assert(len(timesteps) > 0)
            return timesteps[0]

    ## @smproperty.doublevector(name="TimestepValues", repeatable="1", information_only="1") #, 
    # @smproperty.xml("""
    #  <DoubleVectorProperty name="TimestepValues"
    #                         repeatable="1"
    #                         information_only="1">
    #     <TimeStepsInformationHelper />
    #     <Documentation>
    #       This magic property sends time information to the animation
    #       panel.  ParaView will then automatically set up the animation to
    #       visit the time steps defined in the file.
    #     </Documentation>
    #   </DoubleVectorProperty>""")
    
    @smproperty.doublevector(name="TimestepValues", repeatable="1", information_only="1")
    def GetTimestepValues(self):
        print('GetTimestepValues')
        return self._get_timesteps()

    # @smproperty.doublevector(name="Dip", default_values=-90)
    # @smdomain.doublerange(min=-90, max=90)
    def RequestInformation(self, request, inInfo, outInfoVec):
        from vtk import vtkStreamingDemandDrivenPipeline
        print('RequestInformation')
        executive = self.GetExecutive()
        outInfo = outInfoVec.GetInformationObject(0)
        outInfo.Remove(executive.TIME_STEPS())
        outInfo.Remove(executive.TIME_RANGE())
        timesteps = self._get_timesteps()
        for t in timesteps:
            outInfo.Append(executive.TIME_STEPS(), t)
           
        outInfo.Append(executive.TIME_RANGE(), timesteps[0])
        outInfo.Append(executive.TIME_RANGE(), timesteps[-1])

        # info.Set(vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(), (0, 60, 0, 60, 0, 0), 6)
        # # t = [1,2,3,4,5]
        # info.Set(vtkStreamingDemandDrivenPipeline.TIME_STEPS(),
        #        t, len(t))
        # info.Set(vtkStreamingDemandDrivenPipeline.TIME_RANGE(),
        #      [t[0], t[-1]], 2)
        #info.Set(vtk.vtkAlgorithm.CAN_PRODUCE_SUB_EXTENT(), 1) 
        
        return 1

    def FillOutputPortInformation(self, port, info):
        from vtk import vtkDataObject
        if port == 0:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkUnstructuredGrid")
            # info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        else:
            info.Set(vtkDataObject.DATA_TYPE_NAME(), "vtkTable")
        return 1


    def RequestData(self, request, inInfoVec, outInfoVec):
        
        from vtkmodules.vtkCommonDataModel import vtkTable
        
        #from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk.util import numpy_support
        
        from vtk import vtkPolyData, vtkPoints, vtkCellArray, vtkFloatArray, VTK_FLOAT, vtkUnstructuredGrid, vtkUnstructuredGridReader, \
                    vtkStreamingDemandDrivenPipeline, VTK_DOUBLE
        
        import numpy as np
        info = outInfoVec.GetInformationObject(0)
        
        data_time = self._get_update_time(outInfoVec.GetInformationObject(0))
        t = info.Get(vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
        print ("Time request  = %f" % t)
        # Read the geometry 
        
        if self._mesh == None :
            geoReader = vtkUnstructuredGridReader()
    
            geoFileName= "/data/lange/simulations/Project-VT-Simulations/model/VT1/ventricle-vol_fibers.vtk"
            print ("Read Geofile: " + geoFileName)
            geoReader.SetFileName(geoFileName)
            geoReader.Update()
            self._mesh =vtkUnstructuredGrid()
            self._mesh.DeepCopy(geoReader.GetOutput())
        #Get the data 
        data= self.GetTimeFrame(t)
        vtkarr = numpy_support.numpy_to_vtk( data, deep=True, array_type=VTK_DOUBLE )
        vtkarr.SetName("IGB")
        outputCurves = vtkUnstructuredGrid.GetData(outInfoVec, 0)
        outputCurves.ShallowCopy(self._mesh)
        outputCurves.GetPointData().AddArray(vtkarr)
        #t1 = time.time()
        # print ("t1-t0", t1-t0)
        #if data_time is not None:
        #    output.GetInformation().Set(output.DATA_TIME_STEP(), data_time)
        return 1

print ("Hi There")