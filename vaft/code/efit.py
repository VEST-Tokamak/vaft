

def gfile_to_omas(self, ods=None, time_index=0, profile_index=0, allow_derived_data=True):
    """
    translate gEQDSK class to OMAS data structure

    :param ods: input ods to which data is added

    :param time_index: time index to which data is added

    :param allow_derived_data: bool
        Allow data to be drawn from fluxSurfaces, AuxQuantities, etc. May trigger dynamic loading.

    :return: ODS
    """
    if ods is None:
        ods = ODS()

    if self.cocos is None:
        cocosio = self.native_cocos()  # assume native gEQDSK COCOS
    else:
        cocosio = self.cocos

    # delete time_slice before writing, since these quantities all need to be consistent
    if 'equilibrium.time_slice.%d' % time_index in ods:
        ods['equilibrium.time_slice.%d' % time_index] = ODS()

    # write derived quantities from fluxSurfaces
    if self['CURRENT'] != 0.0:
        flx = self['fluxSurfaces']
        ods = flx.to_omas(ods, time_index=time_index)

    eqt = ods[f'equilibrium.time_slice.{time_index}']

    # align psi grid
    psi = np.linspace(self['SIMAG'], self['SIBRY'], len(self['PRES']))
    if f'equilibrium.time_slice.{time_index}.profiles_1d.psi' in ods:
        with omas_environment(ods, cocosio=cocosio):
            m0 = psi[0]
            M0 = psi[-1]
            m1 = eqt['profiles_1d.psi'][0]
            M1 = eqt['profiles_1d.psi'][-1]
            psi = (psi - m0) / (M0 - m0) * (M1 - m1) + m1
    coordsio = {f'equilibrium.time_slice.{time_index}.profiles_1d.psi': psi}

    # add gEQDSK quantities
    with omas_environment(ods, cocosio=cocosio, coordsio=coordsio):

        try:
            ods['dataset_description.data_entry.pulse'] = int(
                re.sub('[a-zA-Z]([0-9]+).([0-9]+).*', r'\1', os.path.split(self.filename)[1])
            )
        except Exception:
            ods['dataset_description.data_entry.pulse'] = 0

        try:
            separator = ''
            ods['equilibrium.ids_properties.comment'] = self['CASE'][0]
        except Exception:
            ods['equilibrium.ids_properties.comment'] = 'omasEQ'

        try:
            # TODO: this removes any sub ms time info and should be fixed
            eqt['time'] = float(re.sub('[a-zA-Z]([0-9]+).([0-9]+).*', r'\2', os.path.split(self.filename)[1])) / 1000.0
        except Exception:
            eqt['time'] = 0.0

        # *********************
        # ESSENTIAL
        # *********************
        if 'RHOVN' in self:  # EAST gEQDSKs from MDSplus do not always have RHOVN defined
            rhovn = self['RHOVN']
        else:

            printd('RHOVN is missing from top level geqdsk, so falling back to RHO from AuxQuantities', topic='OMFITgeqdsk')
            rhovn = self['AuxQuantities']['RHO']

        # ============0D
        eqt['global_quantities.magnetic_axis.r'] = self['RMAXIS']
        eqt['global_quantities.magnetic_axis.z'] = self['ZMAXIS']
        eqt['global_quantities.psi_axis'] = self['SIMAG']
        eqt['global_quantities.psi_boundary'] = self['SIBRY']
        eqt['global_quantities.ip'] = self['CURRENT']

        # ============0D time dependent vacuum_toroidal_field
        ods['equilibrium.vacuum_toroidal_field.r0'] = self['RCENTR']
        ods.set_time_array('equilibrium.vacuum_toroidal_field.b0', time_index, self['BCENTR'])

        # ============1D
        eqt['profiles_1d.f'] = self['FPOL']
        eqt['profiles_1d.pressure'] = self['PRES']
        eqt['profiles_1d.f_df_dpsi'] = self['FFPRIM']
        eqt['profiles_1d.dpressure_dpsi'] = self['PPRIME']
        eqt['profiles_1d.q'] = self['QPSI']
        eqt['profiles_1d.rho_tor_norm'] = rhovn

        # ============2D
        eqt['profiles_2d.0.grid_type.index'] = 1
        eqt['profiles_2d.0.grid.dim1'] = np.linspace(0, self['RDIM'], self['NW']) + self['RLEFT']
        eqt['profiles_2d.0.grid.dim2'] = np.linspace(0, self['ZDIM'], self['NH']) - self['ZDIM'] / 2.0 + self['ZMID']
        eqt['profiles_2d.0.psi'] = self['PSIRZ'].T
        if 'PCURRT' in self:
            eqt['profiles_2d.0.j_tor'] = self['PCURRT'].T

        # *********************
        # DERIVED
        # *********************

        if self['CURRENT'] != 0.0:
            # ============0D
            eqt['global_quantities.magnetic_axis.b_field_tor'] = self['BCENTR'] * self['RCENTR'] / self['RMAXIS']
            eqt['global_quantities.q_axis'] = self['QPSI'][0]
            eqt['global_quantities.q_95'] = interpolate.interp1d(np.linspace(0.0, 1.0, len(self['QPSI'])), self['QPSI'])(0.95)
            eqt['global_quantities.q_min.value'] = self['QPSI'][np.argmin(abs(self['QPSI']))]
            eqt['global_quantities.q_min.rho_tor_norm'] = rhovn[np.argmin(abs(self['QPSI']))]

            # ============1D
            Psi1D = np.linspace(self['SIMAG'], self['SIBRY'], len(self['FPOL']))
            # eqt['profiles_1d.psi'] = Psi1D #no need bacause of coordsio
            eqt['profiles_1d.phi'] = self['AuxQuantities']['PHI']
            eqt['profiles_1d.rho_tor'] = rhovn * self['AuxQuantities']['RHOm']

            # ============2D
            eqt['profiles_2d.0.b_field_r'] = self['AuxQuantities']['Br'].T
            eqt['profiles_2d.0.b_field_tor'] = self['AuxQuantities']['Bt'].T
            eqt['profiles_2d.0.b_field_z'] = self['AuxQuantities']['Bz'].T
            eqt['profiles_2d.0.phi'] = (interp1e(Psi1D, self['AuxQuantities']['PHI'])(self['PSIRZ'])).T

    if self['CURRENT'] != 0.0:
        # These quantities don't require COCOS or coordinate transformation
        eqt['boundary.outline.r'] = self['RBBBS']
        eqt['boundary.outline.z'] = self['ZBBBS']
        if allow_derived_data and 'Rx1' in self['AuxQuantities'] and 'Zx1' in self['AuxQuantities']:
            eqt['boundary.x_point.0.r'] = self['AuxQuantities']['Rx1']
            eqt['boundary.x_point.0.z'] = self['AuxQuantities']['Zx1']
        if allow_derived_data and 'Rx2' in self['AuxQuantities'] and 'Zx2' in self['AuxQuantities']:
            eqt['boundary.x_point.1.r'] = self['AuxQuantities']['Rx2']
            eqt['boundary.x_point.1.z'] = self['AuxQuantities']['Zx2']

    # Set the time array
    ods.set_time_array('equilibrium.time', time_index, eqt['time'])

    # ============WALL
    ods['wall.description_2d.0.limiter.type.name'] = 'first_wall'
    ods['wall.description_2d.0.limiter.type.index'] = 0
    ods['wall.description_2d.0.limiter.type.description'] = 'first wall'
    ods['wall.description_2d.0.limiter.unit.0.outline.r'] = self['RLIM']
    ods['wall.description_2d.0.limiter.unit.0.outline.z'] = self['ZLIM']

    # Set the time array (yes... also for the wall)
    ods.set_time_array('wall.time', time_index, eqt['time'])

    # Set reconstucted current (not yet in m-files)
    ods['equilibrium.time_slice'][time_index]['constraints']['ip.reconstructed'] = self['CURRENT']

    # Store auxiliary namelists
    code_parameters = ods['equilibrium.code.parameters']
    if 'time_slice' not in code_parameters:
        code_parameters['time_slice'] = ODS()
    if time_index not in code_parameters['time_slice']:
        code_parameters['time_slice'][time_index] = ODS()
    if 'AuxNamelist' in self:
        for items in self['AuxNamelist']:
            if '__comment' not in items:  # probably not needed
                code_parameters['time_slice'][time_index][items.lower()] = ODS()
                for item in self['AuxNamelist'][items]:
                    code_parameters['time_slice'][time_index][items.lower()][item.lower()] = self['AuxNamelist'][items.upper()][
                        item.upper()
                    ]

    return ods
