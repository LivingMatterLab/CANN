! ----------------------------------------------------------------------  !
!! Universal UANISOHYPER_INV for constitutive artificial neural networks !!
! ----------------------------------------------------------------------  !
! When using, please cite: 
! "On automated model discovery and a universal material subroutine" 
! M. Peirlinck, K. Linka, J.A. Hurtado, E. Kuhl
!
! ----------------------------------------------------------------------  !
!
      subroutine uanisohyper_inv (aInv, ua, zeta, nFibers, nInv,
     *                            ui1, ui2, ui3, temp, noel,
     *                            cmname, incmpFlag, ihybFlag,
     *                            numStatev, statev,
     *                            numFieldv, fieldv, fieldvInc,
     *                            numProps, props)
c
c     UANISOHYPER_INV for Constitutive ANN
c     - Only fully incompressible response
c      
      include 'aba_param.inc'
      include 'aba_tcs_param.inc'
c
      character *80 cmname
      dimension aInv(nInv), ua(2), zeta(nFibers*(nFibers-1)/2)
      dimension ui1(nInv), ui2(nInv*(nInv+1)/2)
      dimension ui3(nInv*(nInv+1)/2), statev(numStatev)
      dimension fieldv(numFieldv), fieldvInc(numFieldv)
      dimension props(numProps)
c
      dimension aInv0(15)
c
      parameter ( zero = 0.d0, one = 1.d0, three = 3.d0 )
c
c     for table collection, parameter tables, property tables
      character*80 cTableColl(n_tcsC_TC_size)
      dimension jTableColl(n_tcsI_TC_size)
      character*80 ptName
      parameter (maxParams = 180)
      character*80 cParams(maxParams)
      dimension iParamsDataType(maxParams), iParams(maxParams)
      dimension rParams(maxParams)
c
c     Process Network coefficients
c
      ptName = ''
c
      jErrorTC = 0
      jError = 0
      numParams = 0
      numRows = 0
      numNodes = 0
      call queryTableCollection(jTableColl, cTableColl, jErrorTC)
      if (jErrorTC.eq.0) then
         ptName = 'UNIVERSAL_TAB'
         call queryParameterTable(ptName, numParams, numRows, jError)
         if (jError.eq.0) then 
            numNodes = numRows
         end if
      end if
c Initialize strain energy function
      ua(1) = zero
c Initialize array of derivatives
      do kInv = 1, nInv
         indx2 = indx(kInv,kInv)
         ui1(kInv)=zero
         ui2(indx2)=zero
      end do
c
c Set array of invariants in reference configuration, aInv0
      aInv0(1) = three
      aInv0(2) = three
      do kInv = 3, nInv
         aInv0(kInv) = one
      end do
      if (nFibers.gt.1) then
         aInv0(6) = zeta(1)
         aInv0(7) = zeta(1)
         if (nFiber.gt.2) then
            aInv0(10) = zeta(2)
            aInv0(11) = zeta(2)
            aInv0(12) = zeta(3)
            aInv0(13) = zeta(3)
         end if
      end if
c
c Add contribution from each active neuron
c
      do jRow = 1, numNodes
         call getParameterTableRow(ptName, jRow, numParams, 
     *        iParamsDataType, iParams, rParams, cParams, jError)
         kInv = iParams(1)
         kf1  = iParams(2)
         kf2  = iParams(3)
         w1   = rParams(4)
         w2   = rParams(5)
c     Set shifted invariant 
         xInv = aInv(kInv) - aInv0(kInv)
c
c     Add contribution from this Node
         call uCANN(xInv, kf1, w1, kf2, w2, 
     *        ua(1), ui1(kInv), ui2(indx(kInv,kInv)) )
      end do
c
      return
      end
c
c
      subroutine uCANN(xInv, kf1, w1, kf2, w2, UA, UI1, UI2)
c
      include 'aba_param.inc'
c
      parameter (one = 1.d0)
c
c     Process first layer
      w0 = one
      call uCANN_h1(kf1, w0, xInv, f1, df1, ddf1)
c     Process second layer
      call uCANN_h2(kf2, w1, f1, f2, df2, ddf2)
c
      UA = UA + w2*f2
      UI1 = UI1+w2*df2*df1
      UI2 = UI2+w2*(ddf2*df1**2+df2*ddf1)
c
      return
      end

c
      subroutine uCANN_h1(kf, w, x, f, df, ddf)
c
      include 'aba_param.inc'
c
      parameter ( zero = 0.d0, two = 2.d0 )
c
c     Branch based on function type
      if ( kf.eq. 1 ) then
c     f(x) = w*x 
         f = w*x
         df = w
         ddf = zero
      else if ( kf.eq. 2 ) then
c     f(x) = (w*x)^2 
         f = w**2 * x**2
         df = w**2 * two*x 
         ddf = w**2 * two
      else if ( kf.ge. 3 ) then
c     f(x) = (w*x)^kf
         f = w**kf * x**kf
         df = kf * w**kf * x**(kf-1) 
         ddf = kf * (kf-1) * w**kf * x**(kf-2)
      end if 
c     
      return
      end

c
      subroutine uCANN_h2(kf, w, x, f, df, ddf)
c
      include 'aba_param.inc'
c
      parameter ( zero = 0.d0, one = 1.d0 )
c
c     Branch based on function type
      if ( kf.eq. 1 ) then
c     f(x) = w*x 
         f = w*x
         df = w
         ddf = zero
      else if ( kf.eq. 2 ) then
c     f(x) = exp(w*x)-1 
         f = exp(w*x)-one
         df = w * exp(w*x)
         ddf = w * df
      else if ( kf.eq. 3 ) then
c     f(x) = -ln(1-w*x)
         f = -log(one-w*x)
         df = w / (one-w*x)
         ddf = df**2 
      end if 
c      
      return
      end

c
c Maps index from Square to Triangular storage of symmetric matrix 
c
      integer function indx( i, j )
c
      include 'aba_param.inc'
c
      ii = min(i,j)
      jj = max(i,j)
c
      indx = ii + jj*(jj-1)/2
c
      return
      end

