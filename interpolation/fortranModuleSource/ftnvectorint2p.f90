      subroutine ftnvectorint2p(invariable,   &
                                inpressure,   &
                                outpressure,  &
                                outvariable,  &
                                fillvalue,    &
                                linlog,       &
                                nleft,        &
                                nleftpin,     &
                                nleftpout,    &
                                npin,         &
                                npout,        &
                                ierr          & 
                                )
      implicit none

      !**********************
      ! Input variables
      !**********************
      integer, intent(in) :: nleft, nleftpin, nleftpout, npin, npout
      double precision, dimension(nleft,npin), intent(in) :: invariable
      double precision, dimension(nleftpin,npin), intent(in) :: inpressure
      double precision, dimension(nleftpout,npout), intent(in) :: outpressure
      double precision, intent(in) :: fillvalue
      integer, intent(in) :: linlog
      integer, intent(inout) :: ierr

      !**********************
      ! Output variables
      !**********************
      double precision, dimension(nleft,npout), intent(out) :: outvariable

      !**********************
      ! Local variables
      !**********************
      integer,target :: ll,l1
      integer, pointer :: lpin, lpout
      double precision, dimension(npin) :: tempPressure,tempVariable


        !Initialize to a no-error state
        ierr = 0
        !Set a pointer target that has the value 1
        l1 = 1

        !Point lpin and lpout to this target initially
        lpin => l1
        lpout => l1

        !Check whether the leftmost dimension of the
        !pressure variable and the input variable match
        if(nleft == nleftpin)then
          !If so, point the input pressure indexor at ll
          lpin => ll
        !If they don't, then assert that the leftmost dimension
        !is 1 (the error flag is set if not)
        elseif (nleftpin /= 1)then
          ierr = 1
        end if

        !Check whether the leftmost dimension of the
        !pressure variable and the output variable match
        if(nleft == nleftpout)then
          !If so, point the output pressure indexor at ll
          lpout => ll
        !If they don't, then assert that the leftmost dimension
        !is 1 (the error flag is set if not)
        elseif (nleftpout /= 1)then
          ierr = 1
        end if

        !Check if any error flags have been set;
        ! return if so
        if(ierr /= 0)then
          return
        end if

        leftdimloop:  &
        do ll = 1, nleft
          ! Call the NCL-based interpolation routine
          ! Either the input or output pressures can be 1D
          ! or they can have a dimensionality that matches the
          ! respective input/output variables.  
          !
          ! We deal with this by using pointers for the indexors
          ! of the pressure variables, lpin and lpout; if 1D, they
          ! point at a variable that is 1--otherwise they point at
          ! the variable indexor ll.
          call dint2p(  inpressure(lpin,:),   &
                        invariable(ll,:),     &
                        tempPressure,         &
                        tempVariable,         &
                        npin,                 &
                        outpressure(lpout,:), &
                        outvariable(ll,:),    &
                        npout,                &
                        linlog,               &
                        fillvalue,            &
                        ierr )

          !Check if an error was flagged; return if so
          if(ierr /= 0)return
        end do leftdimloop

        return
        
      end subroutine ftnvectorint2p
