from __future__ import division
from scipy.optimize import least_squares
import numpy as np

class MagCalc:
    """Object to calculate magnetic fields in an atomic structure.

    Attributes:
        atoms (ndarray): The positions of the atoms in the crystal structure.
            Should be a 2D numpy array with each row as a 1D numpy array with 3
            values designating an x,y,z location.
        spins (ndarray): The magnetic dipole moments of the atoms in the crystal
            structure. Should be a 2D numpy array with each row as a 1D numpy
            array with 3 values designating an i,j,k spin direction.
        locations (ndarray): The locations of where to calculate the magnetic
            field. Should be a 2D numpy array with each row as a 1D numpy array
            with 3 values designating an x,y,z location.
    """

    def __init__(self,
                 atoms,
                 spins,
                 locations=None,
                 g_factor=1,
                 spin=0.5,
                 magneton='mu_B'):
        """Initializes the spins and atomic positions of a crystal structure.
        Optionally intializes the locations for where to calculate the field.

        Parameters:
            atoms (ndarray): The positions of the atoms in the crystal
                structure. Should be a 2D numpy array with each row as a 1D
                numpy array with 3 values designating an x,y,z location.
            spins (ndarray): The magnetic dipole moments of the atoms in the
                crystal structure. Should be a 2D numpy array with each row as a
                1D numpy array with 3 values designating an i,j,k spin
                direction.
            locations (ndarray): Optional. The locations of where to calculate
                the magnetic field. Should be a 2D numpy array with each row as
                a 1D numpy array with 3 values designating an x,y,z location.
            g_factor (int or float): The dimensionless g-factor used to
                calculate the spin magnetic moment.
            spin (int or float): The spin quantum number.
            magneton (string): Either 'mu_B' or 'mu_N' depending on whether the
                magnetic moment depends on the nuclei or electrons.
        """

        MAGNETONS = {'mu_B':9.274009994e-24, 'mu_N':5.050783699e-27}

        try:
            const = MAGNETONS[magneton]
        except:
            raise ValueError("magneton must be 'mu_B' or 'mu_N'")

        self.g_factor = g_factor
        self.spin = spin
        self.magneton = magneton

        eigenvalue = spin #(spin * (spin+1))**(0.5)
        self.atoms = atoms
        self.spins = spins * g_factor * eigenvalue * const
        self.locations = locations

    def calculate_field(self,
                        location,
                        return_vector=True,
                        mask=None):
        """ Calculates the magnetic field at the specified location.

        Parameters:
            location (ndarray): A 1D numpy array of length 3 specifying where to
                calculate the magnetic field.
            return_vector (boolean): Optional, default is True. See below for
                details.
            mask (ndarray): Optional, default is None. Should be an ndarray
                that is the same size as self.atoms and self.spins.

        Returns:
            (float or ndarray): The magnetic field at the given location. If
                return_vector is False it is a float of the magnitude. If
                return_vector is True, it is a 1D numpy array with 3 values.

        """
        if mask is not None:
            atoms = self.atoms[mask]
            spins = self.spins[mask]
        else:
            atoms = self.atoms
            spins = self.spins

        r = (location - atoms) * 1e-10
        m = spins.copy()

        m_dot_r = (r*m).sum(axis=1)
        r_mag = np.linalg.norm(r, axis=1)

        Bvals = 3.0 * r.T * m_dot_r / r_mag**5 - m.T / r_mag**3
        Btot = Bvals.sum(axis=1) * 1e-7 # (mu_0 / (4 * pi))

        return Btot if return_vector is True else np.linalg.norm(Btot)

    def calculate_fields(self,
                         locations=None,
                         return_vector=True,
                         mask_radius=None):
        """ Calculates the magnetic field at the specified locations.

        Parameters:
            locations (ndarray): Optional. A 2-Dimensional numpy array
                specifying at which locations to calculate the magnetic field.
                Each row in the array should be a 1-Dimensional numpy array of
                length 3. If no value is passed in then it will use
                self.locations.
            return_vector (boolean): Optional, default is True. See below for
                details.
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculations. If mask_radius is set to a number,
                only atoms and spins within mask_radius units of all locations
                will be taken into account. (Only recommended if
                len(locations) > 30. In that case, recommeded value is 8 to
                preserve accuracy.)

        Returns:
            (ndarray): A numpy array of either floats or 1D numpy arrays for the
                magnetic field at each of the given locations. If return_vector
                is False it is a 1D array of floats representing the magnitude.
                If return_vector is True, it is a 2D numpy array of 1D numpy
                arrays with 3 values.

        """

        if locations is None:
            if self.locations is None:
                raise ValueError('Please specify locations first')
            else:
                locations = self.locations

        if locations.ndim == 1:
            locations = np.expand_dims(locations, axis=0)

        if mask_radius is not None:
            mask = self.make_mask(locations, mask_radius)
        else:
            mask = None

        return np.array([self.calculate_field(location,
                                              return_vector,
                                              mask)
                                              for location in locations])


    def find_field(self,
                   field,
                   center_point=np.zeros(3),
                   search_range=10,
                   mask=None):
        """ Finds the location of a magnetic field in the crystal structure
        using least squares minimization.

        Parameters:
            field (float or int): The value of the magnetic field the function
                searches for. (Tesla)
            mask (ndarray): Optional, default is None. Should be an ndarray
                that is the same size as self.atoms and self.spins.

        Returns:
            (ndarray): A 1D array containing the x,y,z location in the
                structure where the magnetic field is closest to the input
                field.

        """
        initial = np.random.rand(3) * search_range + center_point - search_range / 2

        f = lambda x, y, z: (self.calculate_field(location=x,
                                                  return_vector=False,
                                                  mask=y) - z)

        minimum = least_squares(f, initial, args=(mask, field))

        return minimum.x

    def make_plane(self,
                   side_length,
                   resolution,
                   center_point,
                   norm_vec=np.array([0,0,1]),
                   return_vector=False,
                   mask_radius=None):
        """ Calculates the magnetic field over a grid specified by the input
        parameters.

        Parameters:
            side_length (int or float): The side length of the plane in
                Angstroms.
            resolution (int): The number of measurments to take per
                Angstrom.
            center_point (ndarray): A 1D numpy array with 3 values specifying an
                x,y,z location that the grid will be centered on.
            norm_vec (ndarray): A 1D numpy array specifying the normal vector to
                the plane we want to make the grid coordinates over. (i.e. the
                vector [a b c] corresponds to the plane ax + by + cz = 0)
            return_vector (boolean): Optional, default is False. See below for
                details.
            mask_radius (int or float): Optional, default is None. If
                mask_radius is None, all atoms and spins will be taken into
                account for the calculations. If mask_radius is given a value,
                only atoms and spins within mask_radius units of the grid will
                be taken into account. (Only recommended if len(locations) > 30.
                In that case, recommeded value is 8 to preserve accuracy.)

        Returns:
            (ndarray): A numpy array containing the magnetic field at each point
                in the grid. The array will have side_length*resolution columns
                and rows. If return_vector is False, the magnetic fields at each
                location in the array will be magnitudes. If return_vector is
                True, the magnetic fields will be 1D numpy arrays with 3 values.

        """
        v1 = np.array([1,0,0])
        if np.allclose(0, norm_vec[1:]):
            v1 = np.array([0,1,0])

        # This code can be a bit confusing so I will do my best to document it.
        # The orthogonal complement of norm_vec is our desired plane.

        # First we make a matrix that reflects any vector on the orthogonal
        # complement of norm_vec (look up Householder transforms for more info).
        Hnorm = np.eye(3) - (np.outer(norm_vec, norm_vec.T) / norm_vec.T.dot(norm_vec))

        # Next we reflect v1 onto the plane.
        v1 = Hnorm.dot(v1)

        # Now we make a vector that is orthogonal to both norm_vec and v2.
        v2 = np.cross(norm_vec, v1)
        # Now we have two vectors to form an orthogonal basis for our plane. (v1 & v2)

        # We create and normalize Q, making it a transition matrix from our now
        # orthonormal basis to the standard basis.
        Q = np.column_stack((v1, v2, np.zeros_like(v1)))
        Q[:,:2] /= np.linalg.norm(Q[:,:2], axis=0)

        a = np.linspace(-side_length/2, side_length/2, resolution*side_length)
        b = np.linspace(-side_length/2, side_length/2, resolution*side_length)

        A,B = np.meshgrid(a,b)

        locations = np.array([A.ravel(), B.ravel(), np.zeros(A.size)])

        # Multiply locations by Q to get the points into the standard basis.
        locations = Q.dot(locations).T + center_point

        fields = self.calculate_fields(locations=locations,
                                       return_vector=return_vector,
                                       mask_radius=mask_radius)

        return fields.reshape(a.size, b.size)

    # Helper functions and magic methods
    def make_mask(self,
                  locations,
                  mask_radius):
        """ Helper function to make mask for calculate_locations. """

        max_locations = locations.max(axis=0)
        min_locations = locations.min(axis=0)
        mid = (max_locations + min_locations)/2

        max_diff = np.apply_along_axis(np.linalg.norm,
                                       1,
                                       locations - mid).max()
        mask_radius += max_diff

        mask = np.apply_along_axis(np.linalg.norm,
                                   1,
                                   mid - self.atoms) <= mask_radius
        return mask

    def __str__(self):
        """ String representation of MagCalc. """

        output = 'atoms shape:\t {}\n'.format(self.atoms.shape)
        output += 'spins shape:\t {}\n'.format(self.spins.shape)

        if self.locations is None:
            output += 'locations:\t {}\n'.format('None')
        else:
            output += 'locations shape: {}\n'.format(self.locations.shape)

        output += 'g_factor:\t {}\n'.format(self.g_factor)
        output += 'spin:\t\t {}\n'.format(self.spin)
        output += 'magneton:\t {}'.format(self.magneton)

        return output
