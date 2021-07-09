from numpy import *
from .fortranModuleSource.ftnvectorint2p import ftnvectorint2p

def interpolatePressureLevels(  inVariable,                 \
                                inPressure,                 \
                                outPressure,                \
                                fillValue = 1e36,           \
                                useLogPressure = True,      \
                                doExtrapolation = False,    \
                                interpolationDimension = -1,\
                                ):
    """Interpolate a geophysical field with a vertical dimension from one
    pressure grid to another.

    
        This routine uses the fortran routine `dint2p' from NCL 6.1.2 to do the
        actual interpolation for each grid column.

        Input Variables:
        ----------------

            inVariable      :   An arraylike field with a vertical dimension
                                (assumed to be the last dimension unless
                                interpolationDimension is set to something
                                other than -1)

            inPressure      :   The pressure grid for the input field.  If it
                                is a 1D field, then its length must match the
                                length of the dimension
                                `interpolationDimension` in inVariable.
                                Otherwise inPressure must have the same
                                dimensionality as inVariable.

            outPressure     :   The pressure grid for the output field. If it
                                is not a 1D field, then the length of all
                                dimensions except `interpolationDimension` must
                                match the dimension length of inVariable

            fillValue       :   The value that indicates missing data in
                                inVariable

            useLogPressure  :   Flags whether the interpolation is linear in
                                log(inPressure); it is linear in inPressure if
                                False

            doExtrapolation :   Flags whether to extrapolate beyond the bounds
                                of inPressure

            interpolationDimension :    The dimension over which to do
                                        interpolation in inVariable. (It is
                                        assumed to be the last dimension if not
                                        set.)

        Returns:
        --------
        
            A numeric array where all but `interpolationDimension' have the same
            length as inVariable; the dimension `interpolationDimension' will have
            the same length as `interpolationDimension' in outPressure.

    """

    #*********************************
    # Check for array conformance
    #*********************************

    #Check that the input arrays are actually arraylik
    try:
        inVariable = array(inVariable)
    except:
        raise ValueError("inVariable is not arraylike")  

    try:
        inPressure = array(inPressure)
    except:
        raise ValueError("inPressure is not arraylike")  

    try:
        outPressure = array(outPressure)
    except:
        raise ValueError("outPressure is not arraylike")  

    #Get the shape/rank of inVariable
    inVarShape = shape(inVariable)
    inVarRank = len(inVarShape)

    #Get the shape/rank of inPressure
    inPresShape = shape(inPressure)
    inPresRank = len(inPresShape)

    #Get the shape/rank of outPressure
    outPresShape = shape(outPressure)
    outPresRank = len(outPresShape)

    #Check that the shapes/ranks are consistent
    if(inPresShape != inVarShape):
        if(inPresRank != 1):
            raise ValueError("inPressure and inVariable are not conformant. Their shapes don't match, and inPressure is not 1D: shape(inPressure) = {}, shape(inVariable) = {}".format(inPresShape,inVarShape))
        elif(inPresShape[0] != inVarShape[interpolationDimension]):
            raise ValueError("inPressure and inVariable are not conformant. Their shapes don't match. inPressure is 1D, but its length ({}) doesn't match the length of dimension {} in inVariable (shape = {}), which is indicated by interpolationDimension.".format(inPresShape[0],interpolationDimension,inVarShape))


    #Check whether the output pressure is 1D
    if(outPresRank != 1):
        #Check whether their ranks match
        if(outPresRank != inVarRank):
            raise ValueError("outPressure and inVariable are not conformant.  They don't have the same rank: shape(outPressure) = {}, shape(inVariable) = {}".format(outPresShape,inVarShape))
        #If their ranks match, check that lengths of all but the interpolation dimension match
        else:
            #Make lists out of the input variable shape and output pressure shape
            inVarTestShape = list(inVarShape)
            outPresTestShape = list(outPresShape)

            #Now that they are lists, pop the interpolation dimension from the shape list
            inVarTestShape.pop(interpolationDimension)
            outPresTestShape.pop(interpolationDimension)

            #Check that these lists now match
            if(inVarTestShape != outPresTestShape):
                raise ValueError("outPressure and inVariable are not conformant. Their shapes don't match for all non-interpolant dimensions, and outPressure is not 1D: shape(outPressure) = {}, shape(inVariable) = {}".format(outPresShape,inVarShape))


    #*********************************
    # Adapt arguments for passing to
    # ftnvectorint2p()
    #*********************************
    #Set the linlog variable
    #   abs(linlog) == 1 --> linear interpolation 
    #   abs(linlog) != 1 --> log interpolation
    # if linlog is negative, we do extrapolation.
    #The following formula transforms useLogPressure and doExtrapolation
    # into values of linlog that are +/- (1,2)
    linlog = (int(useLogPressure) + 1) * (-1)**int(doExtrapolation)

    #Create a permuted version of the variable shape with the interpolation dimension in the last dimension
    permutedDimensions = roll(list(range(inVarRank)),-(interpolationDimension+1))
    #Get the total size of the leftmost dimensions
    permutedInputShape = roll(inVarShape,-(interpolationDimension+1))
    leftDimensionProduct = product(permutedInputShape[:-1])
    #Use this permutation to transpose the input variable.  Simultaneously reshape it such that it is 2D.
    reshapedInput = transpose(inVariable,permutedDimensions).reshape([leftDimensionProduct,inVarShape[interpolationDimension]])

    #Reshape the input pressure as necessary
    if(inPresRank == inVarRank):
        reshapedInPressure = transpose(inPressure,permutedDimensions).reshape([leftDimensionProduct,inPresShape[interpolationDimension]])
    else:
        reshapedInPressure = inPressure.reshape([1,len(inPressure)])

    #Reshape the output pressure as necessary
    if(outPresRank == inVarRank):
        reshapedOutPressure = transpose(outPressure,permutedDimensions).reshape([leftDimensionProduct,outPresShape[interpolationDimension]])
    else:
        reshapedOutPressure = outPressure.reshape([1,len(outPressure)])

    #*********************************
    # Call ftnvectorint2p(), which
    # is a wrapper for dint2p()
    #*********************************

    ierr = 0
    reshapedOutput = ftnvectorint2p( invariable = reshapedInput,        \
                                     inpressure = reshapedInPressure,   \
                                     outpressure = reshapedOutPressure, \
                                     fillvalue = fillValue,             \
                                     linlog = linlog,                   \
                                     ierr = ierr,                       \
                                   )
    #Check whether any errors were raised
    if(ierr != 0):
        raise RuntimeError("ftnvectorint2p() returned a non-zero error code.")

    #********************************
    # Unpermute the output
    #********************************
    permutedOutputShape = permutedInputShape
    permutedOutputShape[-1] = outPresShape[interpolationDimension]
    permutedDimensions = roll(list(range(inVarRank)),interpolationDimension+1)

    #Reshape the output variable so that its non-vertical dimensions match
    #those of the input variable and undo the transpose operation to put the
    #vertical dimension in the same dimension as the original input variable
    outVariable = transpose(reshapedOutput.reshape(permutedOutputShape),permutedDimensions)

    return outVariable


if __name__ == "__main__" :

    from iliad.utilities import ncl
    import os

    pi =array([1000.,925.,850.,700.,600.,500.,400.,300.,250., \
          200.,150.,100.,70.,50.,30.,20.,10. ])
           
    xi =array([ 28., 23., 18., 10.,  2., -4., -15.,-30.,-40., \
        -52.,-67.,-78.,-72.,-61.,-52.,-48.,-46. ])

    po =array([ 1000.,950.,900.,850.,800.,750.,700.,600.,500., \
          425.,400.,300.,250.,200.,100.,85.,70.,50.,40.,\
           30.,25.,20.,15.,10. ])

    xo = interpolatePressureLevels(xi,pi,po,useLogPressure=True,doExtrapolation=False)

    xoStrings = ["{:12.12f}".format(xval) for xval in xo]
    xoArray = "(/{}/)".format(", ".join(xoStrings))

    nclCode = """
begin
  linlog = 2   ; ln(p) interpolation

  pi =(/ 1000.,925.,850.,700.,600.,500.,400.,300.,250., \
          200.,150.,100.,70.,50.,30.,20.,10. /)
           
  xi =(/ 28., 23., 18., 10.,  2., -4., -15.,-30.,-40., \
        -52.,-67.,-78.,-72.,-61.,-52.,-48.,-46. /)

  po =(/ 1000.,950.,900.,850.,800.,750.,700.,600.,500., \
          425.,400.,300.,250.,200.,100.,85.,70.,50.,40.,\
           30.,25.,20.,15.,10. /)

  xtest = {}
           
; Note: you could use "int2p" here as well.
  xo = int2p_n (pi,xi,po,linlog,0)

  xdiffavg = avg(xo-xtest)
  xdiffmax = max(abs(xo-xtest))
  print("Test fields differed by " + xdiffmax + " at most and " + xdiffavg + " on average")
; xo will contain (/ 28.,24.71,21.37,18. ,...., -48.,-47.17,-46./).

  if(xdiffavg.ne.0.0.or.xdiffmax.ne.0.0)then
    status_exit(1)
  end if

status_exit(0)
end
status_exit(1)

""".format(xoArray)

    nclOutput = ncl.execute(nclCode)

    print(nclOutput)

    #*******************************************************
    # Test that several versions of 3D interpolation work
    #*******************************************************
    #A 3D input variable with 1D input/output pressures
    xi3d = transpose(xi*ones([100,200,len(pi)]),[2,0,1])
    xo3d = interpolatePressureLevels(xi3d,pi,po,useLogPressure=True,doExtrapolation=False,interpolationDimension=0)
    assert( sum(xo3d[:,10,10]-xo) == 0.0)
    print("Test of 3D input variable with 1D input/output pressures successful")

    #A 3D input variable with 3D input pressure and 1D output pressure
    pi3d = transpose(pi*ones([100,200,len(pi)]),[2,0,1])
    xo3d = interpolatePressureLevels(xi3d,pi3d,po,useLogPressure=True,doExtrapolation=False,interpolationDimension=0)
    assert( sum(xo3d[:,10,10]-xo) == 0.0)
    print("Test of 3D input variable with 3D input pressure and 1D output pressure successful")

    #A 3D input variable with 1D input pressure and 3D output pressure
    po3d = transpose(po*ones([100,200,len(po)]),[2,0,1])
    xo3d = interpolatePressureLevels(xi3d,pi,po3d,useLogPressure=True,doExtrapolation=False,interpolationDimension=0)
    assert( sum(xo3d[:,10,10]-xo) == 0.0)
    print("Test of 3D input variable with 1D input pressure and 3D output pressure successful")

    #A 3D input variable with 3D input pressure and 3D output pressure
    po3d = transpose(po*ones([100,200,len(po)]),[2,0,1])
    xo3d = interpolatePressureLevels(xi3d,pi3d,po3d,useLogPressure=True,doExtrapolation=False,interpolationDimension=0)
    assert( sum(xo3d[:,10,10]-xo) == 0.0)
    print("Test of 3D input variable with 3D input pressure and 3D output pressure successful")


    print("All tests successful")

