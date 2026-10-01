! vaft_plasma_state -- build a NUBEAM input Plasma State from public NTCC APIs.
!
! Part of VAFT (MIT licence). This program is VAFT's own and contains no NTCC
! source. The VAFT NUBEAM installers compile it against the user's build of the
! public NTCC NUBEAM distribution (https://w3.pppl.gov/NTCC/NUBEAM/), and it
! calls only the documented Plasma State API in plasma_state_mod.
!
! Usage:  vaft_plasma_state [namelist-file]     (default: vaft_plasma_state.nml)
!
! Input is one namelist, &vaft_plasma_state, written by
! vaft.code.nubeam.plasma_state:
!
!   * the machine description and shot configuration namelists (NTCC's own
!     mdescr / sconfig formats) and a G-EQDSK;
!   * kinetic profiles tabulated at x(1:nx), x(1) = 0 and x(nx) = 1, where x
!     is either "rho_tor" (sqrt of normalised toroidal flux, the Plasma State
!     coordinate) or "sqrt_psi_n" (sqrt of normalised poloidal flux). The
!     latter is mapped onto rho_tor through the equilibrium this program has
!     just loaded, so the profiles and the geometry share one flux surface
!     labelling;
!   * densities in m^-3, temperatures in keV, omegat in rad/s; ion densities
!     follow the thermal-ion order of the sconfig namelist (iZatom_S, iAMU_S);
!   * per-beam power [W] and energy [keV], and the 0-D neutral source data.
!
! Unlike most NTCC drivers this program exits non-zero on every failure, so
! the caller can rely on the exit status.
program vaft_plasma_state_main
  use plasma_state_mod
  implicit none

  integer, parameter :: maxpts = 2001, maxion = 20, maxbeam = 32, maxgas = 16
  integer, parameter :: r8 = rspec

  character(len=512) :: mdescr_file = ' ', sconfig_file = ' ', geqdsk_file = ' '
  character(len=512) :: output_file = 'vaft_plasma_state.cdf'
  character(len=256) :: runid = 'VAFT', global_label = 'vaft_plasma_state'
  character(len=32) :: x_coordinate = 'rho_tor'
  real(r8) :: t0 = 0.0_r8, t1 = 0.0_r8
  real(r8) :: bdy_crat = 0.08_r8
  integer :: kcur_option = -1
  integer :: nmom = 64
  logical :: limiter_from_geqdsk = .true.
  integer :: nrho = 101, nth_eq = 101
  integer :: nx = 0
  real(r8) :: x(maxpts) = 0.0_r8
  real(r8) :: ne(maxpts) = 0.0_r8, te(maxpts) = 0.0_r8, ti(maxpts) = 0.0_r8
  real(r8) :: zeff(maxpts) = 0.0_r8, omegat(maxpts) = 0.0_r8
  real(r8) :: ni(maxpts, maxion) = 0.0_r8
  integer :: nion = 0, nbeam = 0, ngas = 0
  real(r8) :: power_nbi(maxbeam) = 0.0_r8, kvolt_nbi(maxbeam) = 0.0_r8
  real(r8) :: dn0out = 0.0_r8, e0_av(maxgas) = 0.0_r8
  integer :: is_recycling(maxgas) = 1

  namelist /vaft_plasma_state/ mdescr_file, sconfig_file, geqdsk_file, &
       output_file, runid, global_label, x_coordinate, t0, t1, bdy_crat, &
       kcur_option, nmom, limiter_from_geqdsk, nrho, nth_eq, nx, x, ne, te, &
       ti, zeff, omegat, ni, nion, nbeam, ngas, power_nbi, kvolt_nbi, dn0out, &
       e0_av, is_recycling

  character(len=512) :: nml_file
  integer :: ierr, iunit, ios, i, k, nz
  real(r8), allocatable :: xs(:)
  logical :: rotation

  nml_file = 'vaft_plasma_state.nml'
  if (command_argument_count() >= 1) call get_command_argument(1, nml_file)

  open(newunit=iunit, file=trim(nml_file), status='old', action='read', iostat=ios)
  if (ios /= 0) call fail('cannot open namelist file '//trim(nml_file))
  read(iunit, nml=vaft_plasma_state, iostat=ios)
  close(iunit)
  if (ios /= 0) call fail('cannot read &vaft_plasma_state from '//trim(nml_file))

  call check_inputs()
  rotation = any(omegat(1:nx) /= 0.0_r8)

  ! Machine description, then shot configuration: the API requires this order.
  if (limiter_from_geqdsk) then
     call ps_mdescr_read(trim(mdescr_file), ierr, g_filename=trim(geqdsk_file))
  else
     call ps_mdescr_read(trim(mdescr_file), ierr)
  end if
  call check(ierr, 'ps_mdescr_read')
  call ps_sconfig_read(trim(sconfig_file), ierr)
  call check(ierr, 'ps_sconfig_read')

  ! The documented follow-up calls. ps_sconfig_read already builds the merged
  ! lists in current releases, and the merge refuses to run twice.
  call ps_label_species(ierr)
  call check(ierr, 'ps_label_species')
  if (.not. allocated(ps%SA_name)) then
     call ps_merge_species_lists(ierr)
     call check(ierr, 'ps_merge_species_lists')
  end if
  if (.not. allocated(ps%SGAS_name)) then
     call ps_neutral_species(ierr)
     call check(ierr, 'ps_neutral_species')
  end if

  if (ps%nspec_th /= nion) then
     write(*, '(a,i0,a,i0,a)') ' ?vaft_plasma_state: sconfig declares ', &
          ps%nspec_th, ' thermal ion species, but ', nion, ' ion profiles were given'
     call fail('thermal ion species count mismatch')
  end if
  if (ps%nbeam /= nbeam) then
     write(*, '(a,i0,a,i0)') ' ?vaft_plasma_state: mdescr declares nbeam = ', &
          ps%nbeam, ', namelist gives ', nbeam
     call fail('beam count mismatch')
  end if
  if (ps%ngsc0 /= ngas) then
     write(*, '(a,i0,a,i0)') ' ?vaft_plasma_state: sconfig declares ngsc0 = ', &
          ps%ngsc0, ', namelist gives ', ngas
     call fail('neutral gas source count mismatch')
  end if

  ! Grids. The profile grids NUBEAM reads all share the state's rho.
  ps%nrho = nrho
  ps%nrho_nbi = nrho
  ps%nrho_gas = nrho
  ps%nrho_fus = nrho
  ps%nrho_eq = nrho
  ps%nrho_eq_geo = nrho
  ps%nth_eq = nth_eq
  call ps_alloc_plasma_state(ierr)
  call check(ierr, 'ps_alloc_plasma_state')

  call uniform(ps%rho)
  call uniform(ps%rho_nbi)
  call uniform(ps%rho_gas)
  call uniform(ps%rho_fus)
  call uniform(ps%rho_eq)
  call uniform(ps%rho_eq_geo)
  do i = 1, nth_eq
     ps%th_eq(i) = -ps_pi + 2.0_r8*ps_pi*real(i - 1, r8)/real(nth_eq - 1, r8)
  end do

  ps%t0 = t0
  ps%t1 = t1
  ps%RunID = runid
  ps%Global_label = global_label
  ps%eqdsk_file = basename(geqdsk_file)
  ! Fourier moments of the flux-surface shapes; the API's own default is 16.
  ps%nmom = nmom

  call ps_update_equilibrium(ierr, g_filepath=trim(geqdsk_file), &
       bdy_crat=bdy_crat, kcur_option=kcur_option)
  call check(ierr, 'ps_update_equilibrium')
  call ps_mhdeq_derive('everything', ierr)
  call check(ierr, 'ps_mhdeq_derive')

  ! Where each state boundary rho(j) sits on the input coordinate x.
  allocate(xs(nrho))
  select case (trim(x_coordinate))
  case ('rho_tor')
     xs = ps%rho
  case ('sqrt_psi_n')
     ! psipol is tabulated on rho_eq, which is the same uniform grid as rho.
     if (ps%psipol(nrho) == ps%psipol(1)) call fail('equilibrium psipol is flat')
     xs = sqrt(max(0.0_r8, (ps%psipol - ps%psipol(1))/(ps%psipol(nrho) - ps%psipol(1))))
     xs(1) = 0.0_r8
     xs(nrho) = 1.0_r8
  case default
     call fail('x_coordinate must be rho_tor or sqrt_psi_n, got '//trim(x_coordinate))
  end select

  ! Kinetic profiles. State arrays hold zone values (step functions between
  ! boundaries), taken as the mean of the two boundary values; the *_bdy
  ! elements carry the edge.
  nz = nrho - 1
  ps%ns(1:nz, 0) = zone(at(ne))
  ps%Ts(1:nz, 0) = zone(at(te))
  do k = 1, nion
     ps%ns(1:nz, k) = zone(at(ni(:, k)))
     ps%Ts(1:nz, k) = zone(at(ti))
  end do
  ps%Ti(1:nz) = zone(at(ti))
  ps%ni(1:nz) = sum(ps%ns(1:nz, 1:nion), dim=2)
  ps%Zeff(1:nz) = zone(at(zeff))
  ps%Zeff_th(1:nz) = zone(at(zeff))
  if (rotation) ps%omegat(1:nz) = zone(at(omegat))

  ps%ns_bdy(0) = ne(nx)
  do k = 1, nion
     ps%ns_bdy(k) = ni(nx, k)
  end do
  ps%Te_bdy = te(nx)
  ps%Ti_bdy = ti(nx)
  ps%rho_bdy_ns = 1.0_r8
  ps%rho_bdy_Te = 1.0_r8
  ps%rho_bdy_Ti = 1.0_r8
  if (rotation) then
     ps%omegat_bdy = omegat(nx)
     ps%rho_bdy_omegat = 1.0_r8
  end if
  ps%ns_is_input = 1
  ps%Ts_is_input = 1

  if (nbeam > 0) then
     ps%power_nbi(1:nbeam) = power_nbi(1:nbeam)
     ps%kvolt_nbi(1:nbeam) = kvolt_nbi(1:nbeam)
  end if
  ps%dn0out = dn0out
  if (ngas > 0) then
     ps%e0_av(1:ngas) = e0_av(1:ngas)
     ps%is_recycling(1:ngas) = is_recycling(1:ngas)
  end if

  call ps_store_plasma_state(ierr, trim(output_file))
  call check(ierr, 'ps_store_plasma_state')
  write(*, '(a)') ' vaft_plasma_state: wrote '//trim(output_file)

contains

  subroutine check_inputs()
    integer :: j
    if (len_trim(mdescr_file) == 0) call fail('mdescr_file is required')
    if (len_trim(sconfig_file) == 0) call fail('sconfig_file is required')
    if (len_trim(geqdsk_file) == 0) call fail('geqdsk_file is required')
    if (nx < 2 .or. nx > maxpts) call fail('nx out of range')
    if (nrho < 3 .or. nrho > maxpts) call fail('nrho out of range')
    if (nth_eq < 17 .or. nth_eq > maxpts) call fail('nth_eq out of range')
    if (nion < 1 .or. nion > maxion) call fail('nion out of range')
    if (nbeam < 0 .or. nbeam > maxbeam) call fail('nbeam out of range')
    if (ngas < 0 .or. ngas > maxgas) call fail('ngas out of range')
    if (nmom < 1 .or. nmom > 64) call fail('nmom must be in 1..64')
    if (t1 < t0) call fail('t1 must not precede t0')
    if (abs(x(1)) > 1.0e-9_r8 .or. abs(x(nx) - 1.0_r8) > 1.0e-9_r8) &
         call fail('x must run from 0 to 1')
    do j = 2, nx
       if (x(j) <= x(j - 1)) call fail('x must increase strictly')
    end do
    if (any(ne(1:nx) <= 0.0_r8)) call fail('ne must be positive')
    if (any(te(1:nx) <= 0.0_r8)) call fail('te must be positive')
    if (any(ti(1:nx) <= 0.0_r8)) call fail('ti must be positive')
    if (any(zeff(1:nx) < 1.0_r8)) call fail('zeff must be >= 1')
    if (any(ni(1:nx, 1:nion) < 0.0_r8)) call fail('ion densities must be >= 0')
    if (any(power_nbi(1:nbeam) < 0.0_r8)) call fail('power_nbi must be >= 0')
    if (any(kvolt_nbi(1:nbeam) < 0.0_r8)) call fail('kvolt_nbi must be >= 0')
  end subroutine check_inputs

  ! f, tabulated at x(1:nx), evaluated at the state boundaries xs(1:nrho).
  function at(f) result(g)
    real(r8), intent(in) :: f(:)
    real(r8) :: g(nrho)
    integer :: j, m
    m = 1
    do j = 1, nrho
       do while (m < nx - 1 .and. x(m + 1) < xs(j))
          m = m + 1
       end do
       g(j) = f(m) + (f(m + 1) - f(m))*(xs(j) - x(m))/(x(m + 1) - x(m))
    end do
  end function at

  function zone(f) result(z)
    real(r8), intent(in) :: f(:)
    real(r8) :: z(nrho - 1)
    z = 0.5_r8*(f(1:nrho - 1) + f(2:nrho))
  end function zone

  subroutine uniform(g)
    real(r8), intent(out) :: g(:)
    integer :: j, n
    n = size(g)
    do j = 1, n
       g(j) = real(j - 1, r8)/real(n - 1, r8)
    end do
  end subroutine uniform

  function basename(path) result(name)
    character(len=*), intent(in) :: path
    character(len=len(path)) :: name
    integer :: j
    j = max(index(trim(path), '/', back=.true.), index(trim(path), '\', back=.true.))
    name = path(j + 1:)
  end function basename

  subroutine check(code, what)
    integer, intent(in) :: code
    character(len=*), intent(in) :: what
    if (code /= 0) then
       write(*, '(a,i0)') ' ?vaft_plasma_state: '//what//' failed, ierr=', code
       call fail(what)
    end if
  end subroutine check

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    write(*, '(a)') ' ?vaft_plasma_state: '//trim(msg)
    error stop 1
  end subroutine fail

end program vaft_plasma_state_main
