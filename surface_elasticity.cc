/* Authors: Andrew McBride, University of Cape Town,            	*/
/*          Ali Javili, University of Erlangen-Nuremberg, 2013  	*/
/*                                                              	*/
/*    Copyright (C) 2010-2013 by the deal.II authors         		*/
/*                        & Ali Javili and Andrew McBride   		*/
/*                                                                  */
/*    This file is subject to QPL and may not be  distributed       */
/*    without copyright and license information. Please refer       */
/*    to the file deal.II/doc/license.html for the  text  and       */
/*    further information on this license.                          */
/*    Please see http://www.cerecam.uct.ac.za/code/surface_energy/doc/html/index.html */
/*    Accompanying paper submitted to ANS                           */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <map>
#include <limits>
#include <stdlib.h>
#include <time.h>

/**
 * \brief We wrap the problem in its own namespace
 */
namespace Surface_Elasticity
{
  using namespace dealii;

  /**
   * \brief Various structures to handle reading input data from the parameter files.
   */
  namespace Parameters
  {
    /**
     * \brief Parameters associated with the FE system
     */
    struct FESystem
    {
        unsigned int poly_degree;
        unsigned int quad_order;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    FESystem::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
        {
          prm.declare_entry("Polynomial degree", "1", Patterns::Integer(0),
              "Displacement system polynomial order");

          prm.declare_entry("Quadrature order", "2", Patterns::Integer(0),
              "Gauss quadrature order");
        }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
        {
          poly_degree = prm.get_integer("Polynomial degree");
          quad_order = prm.get_integer("Quadrature order");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the geometry
     */
    struct Geometry
    {
        unsigned int global_refinement;
        std::string input_deck;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    Geometry::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
        {
          prm.declare_entry("Global refinement", "2", Patterns::Integer(0),
              "Global refinement level");

          prm.declare_entry("Input deck", "meshes/input.inp",
              Patterns::FileName(), "Input deck");

        }
      prm.leave_subsection();
    }

    void
    Geometry::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
        {
          global_refinement = prm.get_integer("Global refinement");
          input_deck = prm.get("Input deck");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the material model
     */
    struct Materials
    {
        double lambda_volume;
        double mu_volume;

        double lambda_surface;
        double mu_surface;
        double gamma_surface;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    Materials::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties volume");
        {
          prm.declare_entry("Lames first parameter", "10e6", Patterns::Double(),
              "Lames first parameter");

          prm.declare_entry("Shear modulus", "10e6", Patterns::Double(),
              "Shear modulus");

        }
      prm.leave_subsection();

      prm.enter_subsection("Material properties surface");
        {
          prm.declare_entry("Lames first parameter", "0.", Patterns::Double(),
              "Lames first parameter");

          prm.declare_entry("Shear modulus", "0.", Patterns::Double(),
              "Shear modulus");

          prm.declare_entry("Boundary potential", "0.", Patterns::Double(),
              "Boundary potential");
        }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties volume");
        {
          lambda_volume = prm.get_double("Lames first parameter");
          mu_volume = prm.get_double("Shear modulus");
        }
      prm.leave_subsection();

      prm.enter_subsection("Material properties surface");
        {
          lambda_surface = prm.get_double("Lames first parameter");
          mu_surface = prm.get_double("Shear modulus");
          gamma_surface = prm.get_double("Boundary potential");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the linear solver
     */
    struct LinearSolver
    {
        std::string type_lin;
        double tol_lin;
        double max_iterations_lin;
        std::string preconditioner_type;
        double preconditioner_relaxation;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    LinearSolver::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
        {
          prm.declare_entry("Solver type", "CG",
              Patterns::Selection("Direct|CG"),
              "Type of solver used to solve the linear system");

          prm.declare_entry("Residual", "1e-6", Patterns::Double(0.0),
              "Linear solver residual (scaled by residual norm)");

          prm.declare_entry("Max iteration multiplier", "1",
              Patterns::Double(0.0),
              "Linear solver iterations (multiples of the system matrix size)");

          prm.declare_entry("Preconditioner type", "jacobi",
              Patterns::Selection("jacobi|ssor"), "Type of preconditioner");

          prm.declare_entry("Preconditioner relaxation", "0.65",
              Patterns::Double(0.0), "Preconditioner relaxation value");
        }
      prm.leave_subsection();
    }

    void
    LinearSolver::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
        {
          type_lin = prm.get("Solver type");
          tol_lin = prm.get_double("Residual");
          max_iterations_lin = prm.get_double("Max iteration multiplier");
          preconditioner_type = prm.get("Preconditioner type");
          preconditioner_relaxation = prm.get_double(
              "Preconditioner relaxation");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the Newton scheme
     */
    struct NonlinearSolver
    {
        unsigned int max_iterations_NR;
        double tol_residual;
        double tol_soln;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    NonlinearSolver::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
        {
          prm.declare_entry("Max iterations Newton-Raphson", "10",
              Patterns::Integer(0),
              "Number of Newton-Raphson iterations allowed");

          prm.declare_entry("Tolerance residual", "1.0e-9",
              Patterns::Double(0.0), "Residual tolerance");

          prm.declare_entry("Tolerance solution", "1.0e-10",
              Patterns::Double(0.), "Solution error tolerance");
        }
      prm.leave_subsection();
    }

    void
    NonlinearSolver::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
        {
          max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
          tol_residual = prm.get_double("Tolerance residual");
          tol_soln = prm.get_double("Tolerance solution");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with postprocessing the data
     */
    struct Postprocessing
    {
        bool postprocess_qp_data;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    Postprocessing::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocessing");
        {
          prm.declare_entry("Quadrature point postprocessing", "false",
              Patterns::Bool(),
              "Write the data stored at the quadrature points");
        }
      prm.leave_subsection();
    }

    void
    Postprocessing::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocessing");
        {
          postprocess_qp_data = prm.get_bool("Quadrature point postprocessing");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the simulation duration and time discretization
     */
    struct Time
    {
        double delta_t;
        double end_time;
        double pre_load_time;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    Time::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
        {
          prm.declare_entry("End time", "1.", Patterns::Double(0.), "End time");

          prm.declare_entry("Pre load time", "0.", Patterns::Double(0.),
              "Pre load time");

          prm.declare_entry("Time step size", "0.1", Patterns::Double(0.),
              "Time step size");
        }
      prm.leave_subsection();
    }

    void
    Time::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
        {
          end_time = prm.get_double("End time");
          pre_load_time = prm.get_double("Pre load time");
          delta_t = prm.get_double("Time step size");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Parameters associated with the boundary conditions
     */
    struct BoundaryConditions
    {
        double final_displacement;
        std::string problem_description;
        double traction_x, traction_y, traction_z;
        int neumann_surface_id;

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    void
    BoundaryConditions::declare_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("BoundaryConditions");
        {
          prm.declare_entry("Final displacement", "0.", Patterns::Double(),
              "Final displacement");

          prm.declare_entry("traction x", "0.", Patterns::Double(),
              "x component of prescribed Neumann traction");

          prm.declare_entry("traction y", "0.", Patterns::Double(),
              "y component of prescribed Neumann traction");

          prm.declare_entry("traction z", "0.", Patterns::Double(),
              "z component of prescribed Neumann traction");

          prm.declare_entry("Neumann surface ID", "-1", Patterns::Integer(-1),
              "Neumann surface ID");

          prm.declare_entry("Problem description", "nanowire",
              Patterns::Selection("nanowire|bridge|rough_surface|cooks"),
              "Problem description");
        }
      prm.leave_subsection();
    }

    void
    BoundaryConditions::parse_parameters (
        ParameterHandler &prm)
    {
      prm.enter_subsection("BoundaryConditions");
        {
          final_displacement = prm.get_double("Final displacement");
          problem_description = prm.get("Problem description");
          traction_x = prm.get_double("traction x");
          traction_y = prm.get_double("traction y");
          traction_z = prm.get_double("traction z");
          neumann_surface_id = prm.get_integer("Neumann surface ID");
        }
      prm.leave_subsection();
    }

    /**
     * \brief Container for all uses-specified parameters
     */
    struct AllParameters : public FESystem,
        public Geometry,
        public Materials,
        public LinearSolver,
        public NonlinearSolver,
        public Postprocessing,
        public Time,
        public BoundaryConditions

    {
        AllParameters (
            const std::string & input_file);

        static void
        declare_parameters (
            ParameterHandler &prm);

        void
        parse_parameters (
            ParameterHandler &prm);
    };

    AllParameters::AllParameters (
        const std::string & input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.read_input(input_file);
      parse_parameters(prm);
    }

    void
    AllParameters::declare_parameters (
        ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Postprocessing::declare_parameters(prm);
      Time::declare_parameters(prm);
      BoundaryConditions::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters (
        ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Postprocessing::parse_parameters(prm);
      Time::parse_parameters(prm);
      BoundaryConditions::parse_parameters(prm);
    }
  }

  /**
   * \brief Series of additional tensor algebra tools.
   */
  namespace AdditionalTools
  {
    /**
     * \f$ C_{ij} = A_{ijkl} B_{kl} \f$
     * @param AA \f$ \mathcal{A} \f$
     * @param B \f$ \mathbf{B} \f$
     * @return \f$ \mathbf{C} \f$
     */
    template <int spacedim>
      Tensor<2, spacedim>
      contract_ijkl_kl (
          const Tensor<4, spacedim>& AA, const Tensor<2, spacedim>& B)
      {
        Tensor<2, spacedim> AA_ijkl_B_kl;
        for (unsigned int i = 0; i < spacedim; ++i)
          for (unsigned int j = 0; j < spacedim; ++j)
            for (unsigned int k = 0; k < spacedim; ++k)
              for (unsigned int l = 0; l < spacedim; ++l)
                AA_ijkl_B_kl[i][j] += AA[i][j][k][l] * B[k][l];
        return AA_ijkl_B_kl;
      }

    /**
     * \f$ C_{ijkl} = A_{ij} B_{kl} \f$
     * @param A \f$ \mathbf{A} \f$
     * @param B \f$ \mathbf{B} \f$
     * @return \f$ \mathcal{C} \f$
     */
    template <int spacedim>
      Tensor<4, spacedim>
      outer_product_ijkl (
          const Tensor<2, spacedim> & A, const Tensor<2, spacedim> & B)
      {
        Tensor<4, spacedim> A_ij_B_kl;

        for (unsigned int i = 0; i < spacedim; ++i)
          for (unsigned int j = 0; j < spacedim; ++j)
            for (unsigned int k = 0; k < spacedim; ++k)
              for (unsigned int l = 0; l < spacedim; ++l)
                A_ij_B_kl[i][j][k][l] += A[i][j] * B[k][l];
        return A_ij_B_kl;
      }

    /**
     * \f$ C_{ijkl} = A_{ik} B_{jl} \f$
     * @param A \f$ \mathbf{A} \f$
     * @param B \f$ \mathbf{B} \f$
     * @return \f$ \mathcal{C} \f$
     */
    template <int spacedim>
      Tensor<4, spacedim>
      outer_product_ikjl (
          const Tensor<2, spacedim> & A, const Tensor<2, spacedim> & B)
      {
        Tensor<4, spacedim> A_ik_B_jl;
        for (unsigned int i = 0; i < spacedim; ++i)
          for (unsigned int j = 0; j < spacedim; ++j)
            for (unsigned int k = 0; k < spacedim; ++k)
              for (unsigned int l = 0; l < spacedim; ++l)
                A_ik_B_jl[i][j][k][l] += A[i][k] * B[j][l];
        return A_ik_B_jl;
      }

    /**
     * \f$ C_{ijkl} = A_{il} B_{jk} \f$
     * @param A \f$ \mathbf{A} \f$
     * @param B \f$ \mathbf{B} \f$
     * @return \f$ \mathcal{C} \f$
     */
    template <int spacedim>
      Tensor<4, spacedim>
      outer_product_iljk (
          const Tensor<2, spacedim> & A, const Tensor<2, spacedim> & B)
      {
        Tensor<4, spacedim> A_il_B_jk;
        for (unsigned int i = 0; i < spacedim; ++i)
          for (unsigned int j = 0; j < spacedim; ++j)
            for (unsigned int k = 0; k < spacedim; ++k)
              for (unsigned int l = 0; l < spacedim; ++l)
                A_il_B_jk[i][j][k][l] += A[i][l] * B[j][k];
        return A_il_B_jk;
      }

    /**
     * \brief Some often used tensors
     */
    template <int spacedim>
      class StandardTensors
      {
        public:

          static const SymmetricTensor<2, spacedim> I;
          static const Tensor<4, spacedim> IxI_ikjl;
          static const Tensor<1, spacedim> zero_vec;

      };

    template <int spacedim>
      const SymmetricTensor<2, spacedim> StandardTensors<spacedim>::I =
          unit_symmetric_tensor<spacedim>();

    template <int spacedim>
      const Tensor<4, spacedim> StandardTensors<spacedim>::IxI_ikjl =
          outer_product_ikjl(
              Tensor<2, spacedim>(
                  AdditionalTools::StandardTensors<spacedim>::I),
              Tensor<2, spacedim>(
                  AdditionalTools::StandardTensors<spacedim>::I));

    template <int spacedim>
      const Tensor<1, spacedim> StandardTensors<spacedim>::zero_vec = Tensor<1,
          spacedim>();
  }

  /**
   * \brief Simple simulation time management class
   */
  class Time
  {
    public:
      /**
       * Constructor that assumes a constant time-step size.
       * @param time_end end time
       * @param pre_load_time the time over which a preload is imposed (surface tension acts a preload)
       * @param delta_t time-step size
       */
      Time (
          const double time_end, const double pre_load_time,
          const double delta_t)
          :
              timestep(0),
              n_timesteps(static_cast<int>((time_end / delta_t) + 0.5)),
              time_current(0.),
              time_end(time_end),
              pre_load_time(pre_load_time),
              delta_t(delta_t)
      {
      }

      virtual
      ~Time ()
      {
      }

      /**
       * Returns the current time.
       * @return current time
       */
      double
      current () const
      {
        return time_current;
      }

      /**
       * Returns the total simulated time
       * @return total simulated time
       */
      double
      end () const
      {
        return time_end;
      }

      /**
       * Returns the time step duration
       * @return time step duration
       */
      double
      get_delta_t () const
      {
        return delta_t;
      }

      /**
       * Returns the current time step number
       * @return the current time step
       */
      unsigned int
      get_timestep () const
      {
        return timestep;
      }

      /**
       * Calculate the ratio (less than 1) of current time to preload time
       * @return the ratio
       */
      double
      get_pre_load_factor () const
      {
        const double ratio = time_current / pre_load_time;
        return ((ratio < 1.0) ? ratio : 1.0);
      }

      /**
       * Increment the time data to next step
       */
      void
      increment ()
      {
        time_current += delta_t;
        ++timestep;
      }

      /**
       * Determines if the simulation is complete
       * @return is the simulation complete?
       */
      bool
      is_simulation_complete ()
      {
        return (timestep <= n_timesteps);
      }

    private:
      unsigned int timestep;
      unsigned int n_timesteps;
      double time_current;
      const double time_end;
      const double pre_load_time;
      const double delta_t;
  };

  /**
   * \brief Controls the response at the level of the quadrature point
   *
   * A ContinuumPoint stores kinematic data relating to the deformation.
   * It also computes the kinetics based on the constitutive relations.
   * A continuum point can exist in the volume or on the surface.
   * Depending on the domain_type, the data represent full-rank quantities
   * in the volume or potentially rank-deficient quantities on the surface.
   */
  template <int spacedim>
    class ContinuumPoint
    {
      public:

        /**
         * \brief Default constructor
         */
        ContinuumPoint ()
            :
                J(1.),
                psi(0.),
                lambda(0.),
                mu(0.),
                gamma_final(0.),
                gamma_current(0.)
        {
          domain_type = invalid;
        }

        /**
         * \brief Constructor that allocates the point to the volume or the surface
         * @param v_or_s Set to 0 for a point in volume and 1 for the surface
         */
        ContinuumPoint (
            const int v_or_s)
            :
                J(1.),
                psi(0.),
                lambda(0.),
                mu(0.),
                gamma_final(0.),
                gamma_current(0.)
        {
          if (v_or_s == 0)
            domain_type = volume;
          else if (v_or_s == 1)
            domain_type = surface;
          else
            ExcInvalidState();
        }

        /**
         * Initialize data for a point in the volume
         * @param X_coords material placement of the the point \f$\mathbf{X}\f$
         * @param lambda_in first Lame modulus \f$\lambda\f$
         * @param mu_in  shear modulus \f$\mu\f$
         */
        void
        initialise_data (
            const Point<spacedim>& X_coords, const double lambda_in,
            const double mu_in)
        {

          Assert(domain_type == volume, ExcInternalError());

            {
              // check the validity of the material properties
              double poisson_ratio = lambda_in / (2.0 * (lambda_in + mu_in));
              AssertThrow(poisson_ratio >= 0 && poisson_ratio < 0.5,
                  ExcIndexRangeType<double>(poisson_ratio, 0, 0.5));
            }

          coordinates_material = X_coords;
          lambda = lambda_in;
          mu = mu_in;
          gamma_final = 0.;

          F = AdditionalTools::StandardTensors<spacedim>::I;
          f = AdditionalTools::StandardTensors<spacedim>::I;
          I_hat = AdditionalTools::StandardTensors<spacedim>::I;
          i_hat = AdditionalTools::StandardTensors<spacedim>::I;

          update_data();
        }

        /**
         * \brief Initialise the data associated with a point on the surface
         * @param X_coords material placement of the the point \f$\mathbf{X}\f$
         * @param lambda_in first Lame modulus \f$\widehat{\lambda}\f$
         * @param mu_in shear modulus \f$\widehat{\mu}\f$
         * @param gamma_in final value for surface tension \f$\widehat{\gamma}\f$
         * @param load_factor surface tension ramping factor
         * @param Identity material identity tensor
         * @param normal
         */
        void
        initialise_data (
            const Point<spacedim>& X_coords, const double lambda_in,
            const double mu_in, const double gamma_in, const double load_factor,
            const Tensor<2, spacedim>& Identity,
            const Tensor<1, spacedim>& normal)
        {
          Assert(domain_type == surface, ExcInternalError());
          gamma_final = gamma_in;
          gamma_current = load_factor * gamma_final;
          normal_material = normal;
          normal_spatial = normal;

          coordinates_material = X_coords;
          lambda = lambda_in;
          mu = mu_in;

          F = Identity;
          f = Identity;
          I_hat = Identity;
          i_hat = Identity;

          update_data();
        }

        /**
         * \brief Update the data at the continuum point
         * @param F_in deformation gradient \f$\mathbf{F}\f$
         * @param f_in inverse deformation gradient \f$\mathbf{f}\f$
         * @param J_in Jacobian of the tangent map between material and spatial configurations
         * @param identity spatial identity tensor
         * @param normal spatial normal
         */
        void
        update_data (
            const Tensor<2, spacedim>& F_in, const Tensor<2, spacedim>& f_in,
            const double J_in, const Tensor<2, spacedim>& identity =
                AdditionalTools::StandardTensors<spacedim>::I,
            const Tensor<1, spacedim>& normal =
                AdditionalTools::StandardTensors<spacedim>::zero_vec)
        {
          F = F_in;
          f = f_in;
          J = J_in;
          i_hat = identity;

          // some sanity checks on the inverse of the rank deficient tensors
          AssertThrow(J > 0, ExcLowerRange(J, 0));
          Assert((f * F - I_hat).norm() < 1e-12, ExcInternalError());
          Assert((F * f - i_hat).norm() < 1e-12, ExcInternalError());

          if (domain_type == surface)
            normal_spatial = normal;

          update_data();
        }

        /**
         * Set the data that stays fixed during the time step
         * @param pre_load_factor
         */
        void
        set_data_each_time_step (
            const double pre_load_factor)
        {
          if (domain_type == surface)
            gamma_current = gamma_final * pre_load_factor;
          else if (domain_type == volume)
            gamma_current = 0.;

          update_data();
        }

        /**
         * \brief return the material normal \f$\mathbf{N}\f$
         * @return material normal
         */
        Tensor<1, spacedim>
        get_normal_material () const
        {
          return normal_material;
        }

        /**
         * \brief return the spatial normal \f$\mathbf{n}\f$
         * @return spatial normal
         */
        Tensor<1, spacedim>
        get_normal_spatial () const
        {
          return normal_spatial;
        }

        /**
         * \brief returns the deformation gradient \f$\mathbf{F}\f$
         * @return the deformation gradient
         */
        Tensor<2, spacedim>
        get_F () const
        {
          return F;
        }

        /**
         * \brief returns the determinant of the tangent map
         * @return Jacobian determinant
         */
        double
        get_J () const
        {
          return J;
        }

        /**
         * Return the free energy
         * @return the Helholtz energy
         */
        double
        get_psi () const
        {
          return psi;
        }

        /**
         * \brief returns the first Piola-Kirchhoff stress \f$\mathbf{P}\f$
         * @return the first Piola-Kirchhoff stress
         */
        Tensor<2, spacedim>
        get_P () const
        {
          return P;
        }

        /**
         * \brief Returns the Cauchy stress \f$\mathbf{\sigma}\f$
         * @return the Cauchy stress
         */
        Tensor<2, spacedim>
        get_sigma () const
        {
          return sigma;
        }

        /**
         * Returns the material tangent \f$ \partial \mathbf{P} / \partial \mathbf{F}\f$
         * @return the tangent
         */
        Tensor<4, spacedim>
        get_AA () const
        {
          return AA;
        }

      protected:

        // location and normal
        Point<spacedim> coordinates_material; /// material placement
        Tensor<1, spacedim> normal_material; /// normal in the material configuration
        Tensor<2, spacedim> I_hat; /// material identity tensor
        Tensor<2, spacedim> i_hat; /// spatial identity tensor
        Tensor<1, spacedim> normal_spatial; /// normal in the spatial configuration

        // kinematics
        Tensor<2, spacedim> F; /// deformation gradient
        Tensor<2, spacedim> f; /// inverse deformation gradient
        double J; /// Jacobian determinant
        double psi; /// free energy

        // kinetics
        Tensor<2, spacedim> P; /// Piola_kirchhoff stress
        Tensor<2, spacedim> sigma; /// Cauchy stress
        Tensor<4, spacedim> AA; /// Piola tangent

        // constitutive parameters
        double lambda, mu, gamma_final, gamma_current;

        /// The ContinuumPoint is either in the volume or on the surface
        enum domain_type_enum
        {
          volume,
          surface,
          invalid
        } domain_type;

        /**
         * \brief Given an updated kinematic state, update the kinetics
         */
        void
        update_data ()
        {
          const double ln_J = std::log(J);
          const Tensor<2, spacedim> f_t = transpose(f);
          const Tensor<2, spacedim> F_t = transpose(F);
          compute_psi(ln_J);
          compute_P(f_t, ln_J);
          compute_sigma(F_t);
          compute_AA(f_t, ln_J);
        }

        /**
         * Compute and update the Helmholtz free energy
         * @param ln_J Logarithm of the Jacobian determinant
         */
        void
        compute_psi (
            const double ln_J)
        {
          double d = 0;
          if (domain_type == volume)
            d = 3.;
          else if (domain_type == surface)
            d = 2.;

          psi = 0.5 * lambda * std::pow(ln_J, 2)
              + 0.5 * mu * (double_contract(F, F) - d - 2. * ln_J)
              + gamma_current * J;

        }

        /**
         * \brief Compute the first Piola-Kirchhoff stress \f$\mathbf{P}\f$
         * @param f_t Transpose of the inverse defomation gradient \f$\mathbf{f}^t\f$
         * @param ln_J Logarithm of the Jacobian determinant
         */
        void
        compute_P (
            const Tensor<2, spacedim>& f_t, const double ln_J)
        {
          // note how for J=1, P!=0
          P = (gamma_current * J + lambda * ln_J - mu) * f_t + mu * F;
        }

        /**
         * \brief Compute the Cauchy stress \f$\mathbf{\sigma}\f$ as the push forward of the first Piola-Kirchhoff stress \f$\mathbf{P}\f$
         * @param F_t Transpose of the defomation gradient \f$\mathbf{F}^t\f$
         */
        void
        compute_sigma (
            const Tensor<2, spacedim>& F_t)
        {
          sigma = (1. / J) * (P * F_t);
          if (domain_type == volume)
            symmetrize(sigma);
        }

        /**
         * \brief Compute the material tangent \f$ \mathcal{A} = \partial \mathbf{P} / \partial \mathbf{F} \f$
         * @param f_t f_t Transpose of the inverse defomation gradient \f$\mathbf{f}^t\f$
         * @param ln_J ln_J Logarithm of the Jacobian determinant
         */
        void
        compute_AA (
            const Tensor<2, spacedim>& f_t, const double ln_J)
        {

          const Tensor<4, spacedim> f_t_otimes_f_t =
              AdditionalTools::outer_product_ijkl(f_t, f_t);

          Tensor<4, spacedim> II;
          Tensor<4, spacedim> DD;

          if (domain_type == volume)
            {
              II = AdditionalTools::StandardTensors<spacedim>::IxI_ikjl;

              DD = -AdditionalTools::outer_product_iljk(f_t, f);
            }
          else if (domain_type == surface)
            {
              const Tensor<2, spacedim> i(
                  AdditionalTools::StandardTensors<spacedim>::I);

              II = AdditionalTools::outer_product_ikjl(i, I_hat);

              DD = -AdditionalTools::outer_product_iljk(f_t, f);

              DD += AdditionalTools::outer_product_ikjl(i - i_hat, f * f_t);
            }

          AA = (lambda + gamma_current * J) * f_t_otimes_f_t + mu * II
              + (gamma_current * J + lambda * ln_J - mu) * DD;
        }
    };

  /**
   * \brief Represent a solid body composed of a volume and an energetic surfaces
   */
  template <int spacedim>
    class Solid
    {
      public:
        /**
         * \brief Constructor
         * @param input_file Parameter file name
         */
        Solid (
            const std::string & input_file);

        /**
         * \brief Basic destructor
         */
        ~Solid ();

        /**
         * \brief Control method that runs the problem
         */
        void
        run ();

      private:

        static const unsigned int dim = spacedim - 1; /// dim dimensional surface embedded in spacedim space

        // structures associated with TBB

        // assemble the system matrix and the RHS
        struct PerTaskData_Assemble_Volume;
        struct ScratchData_Assemble_Volume;
        struct PerTaskData_Assemble_Surface;
        struct ScratchData_Assemble_Surface;

        // update the CP data
        struct PerTaskData_UCP_Volume;
        struct ScratchData_UCP_Volume;
        struct PerTaskData_UCP_Surface;
        struct ScratchData_UCP_Surface;

        /**
         * initialise data at the beginning of a time-step
         */
        void
        set_data_beginning_time_step ();

        /**
         * \brief Create the mesh
         */
        void
        make_grid ();

        /**
         * \brief Setups the system matrix, sparsity pattern and extracts the surface mesh
         */
        void
        system_setup ();

        /**
         * \brief Managers the TBB assembly of the contributions from the volume to system matrix
         */
        void
        assemble_system_volume ();

        /**
         *  \brief Assemble the contribution from a cell in volume to the system matrix and RHS
         * @param cell The current cell
         * @param scratch Scratch data
         * @param data Per-task data
         */
        void
        assemble_system_one_cell_volume (
            const typename DoFHandler<spacedim>::active_cell_iterator & cell,
            ScratchData_Assemble_Volume & scratch,
            PerTaskData_Assemble_Volume & data);

        /**
         * \brief Copy the local system matrix and RHS from a cell in the volume to the global matrix and RHS associated with the Solid
         * @param data Per-task data
         */
        void
        copy_local_to_global_volume (
            const PerTaskData_Assemble_Volume & data);

        /**
         * \brief Managers the TBB assembly of the contributions from the surface to system matrix and RHS
         */
        void
        assemble_system_surface ();

        /**
         * \brief Assemble the contribution from a cell on surface to the system matrix and RHS
         * @param cell The current cell
         * @param scratch Scratch data
         * @param data Per-task data
         */
        void
        assemble_system_one_cell_surface (
            const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
            ScratchData_Assemble_Surface & scratch,
            PerTaskData_Assemble_Surface & data);

        /**
         * \brief Copy the local system matrix and RHS from a cell on the surface to the global matrix and RHS associated with the Solid
         * @param data Per-task data
         */
        void
        copy_local_to_global_surface (
            const PerTaskData_Assemble_Surface & data);

        /**
         * Construct the Dirichlet boundary conditions
         * @param it_nr The current iteration in the Newton-Raphson scheme
         */
        void
        make_constraints (
            const int & it_nr);

        /**
         * \brief Construct the map between a cell in the volume and vector containing its ContinuumPoint
         */
        void
        setup_cp_volume ();

        /**
         * \brief Construct the map between a cell on the surface and vector containing its ContinuumPoint
         */
        void
        setup_cp_surface ();

        /**
         * \brief Managers TBB update of the ContinuumPoint in the volume
         * @param solution_delta Accumulated change of the solution within the time step
         */
        void
        update_volume_cp_incremental (
            const Vector<double> & solution_delta);

        /**
         * \brief Managers TBB update of the ContinuumPoint on the surface
         * @param solution_delta Accumulated change of the solution within the time step
         */
        void
        update_surface_cp_incremental (
            const Vector<double> & solution_delta);

        /**
         * \brief Updates the data of a volume ContinuumPoint in the cell using the current deformation state
         * @param cell The current cell
         * @param scratch Scratch data
         * @param data Per-task data
         */
        void
        update_volume_cp_incremental_one_cell (
            const typename DoFHandler<spacedim>::active_cell_iterator & cell,
            ScratchData_UCP_Volume & scratch, PerTaskData_UCP_Volume & data);

        /**
         * \brief Updates the data of a surface ContinuumPoint in the cell using the current deformation state
         * @param cell The current cell
         * @param scratch Scratch data
         * @param data Per-task data
         */
        void
        update_surface_cp_incremental_one_cell (
            const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
            ScratchData_UCP_Surface & scratch, PerTaskData_UCP_Surface & data);

        /**
         * \brief Void function but required by TBB
         * @param data Per-task data
         */
        void
        copy_local_to_global_ucp_volume (
            const PerTaskData_UCP_Volume & data)
        {
        }

        /**
         * \brief Void function but required by TBB
         * @param data Per-task data
         */
        void
        copy_local_to_global_ucp_surface (
            const PerTaskData_UCP_Surface & data)
        {
        }

        /**
         * \brief Controller for the Newton-Raphson solution of the nonlinear problem
         * @param solution_delta The change in the solution during the time step
         */
        void
        solve_nonlinear_timestep (
            Vector<double> & solution_delta);

        /**
         * \brief Solve the linearised system
         * @param newton_update The incremental change in the solution
         * @return The number of iterations and the final residual required by linear solver
         */
        std::pair<unsigned int, double>
        solve_linear_system (
            Vector<double> & newton_update);

        /**
         * \brief Computes the current solution in volume from the value at the beginning of the time step and the accumulated change
         * @param solution_delta The change in the solution during the time step
         * @return The current solution
         */
        Vector<double>
        get_total_solution_volume (
            const Vector<double> & solution_delta) const;

        /**
         * \brief Computes the current solution on the surface from the value at the beginning of the time step and the accumulated change
         * @param solution_delta The change in the solution during the time step
         * @return The current solution
         */
        Vector<double>
        get_total_solution_surface (
            const Vector<double> & solution_delta);

        /**
         * \brief Given a set of data associated with the volume, extract the values associated with dof on the surface
         * @param volume_data The data in the volume
         * @return The data restricted to the surface
         */
        Vector<double>
        extract_surface_data (
            const Vector<double> & volume_data);

        // ToDo: make constant function
        /**
         * \brief Output the results for post-processing
         */
        void
        output_results ();

        /**
         * \brief Create a map between the surface and volume dof
         */
        void
        setup_surface_volume_dof_map ();

        Parameters::AllParameters parameters; /// simulation parameters

        double vol_reference; /// volume of the reference configuration
        double vol_spatial; /// volume of the spatial configuration

        Triangulation<spacedim> volume_triangulation; /// Triangulation of the volume
        Triangulation<dim, spacedim> surface_triangulation; /// Triangulation of the surface

        Time the_time;
        TimerOutput timer; // basic performance evaluation

        /// map between a cell in volume and the ContinuumPoint data
        std::map<typename Triangulation<spacedim>::cell_iterator,
            std::vector<ContinuumPoint<spacedim> > > volume_cp_map;

        /// map between a cell on the surface and the ContinuumPoint data
        std::map<typename Triangulation<dim, spacedim>::cell_iterator,
            std::vector<ContinuumPoint<spacedim> > > surface_cp_map;

        const unsigned int poly_order; /// order of the polynomial interpolation

        MappingQ<spacedim> volume_mapping_ref; /// to map the material volume mesh to spatial configuration
        MappingQ<dim, spacedim> surface_mapping_ref; /// to map the material surface mesh to spatial configuration

        // independent surface and volume FESystems
        const FESystem<spacedim> volume_fe;
        const FESystem<dim, spacedim> surface_fe;

        // independent surface and volume DoFHandlers
        DoFHandler<spacedim> volume_dof_handler;
        DoFHandler<dim, spacedim> surface_dof_handler;

        // number of degrees of freedom in the volume and on the surface
        const unsigned int dofs_per_volume_cell;
        const unsigned int dofs_per_surface_cell;

        // extract the dof as a vector
        const FEValuesExtractors::Vector u_fe;
        static const unsigned int n_u_components = spacedim;
        static const unsigned int first_u_component = 0;

        // quadrature related data
        const QGauss<spacedim> qf_volume;
        const QGauss<dim> qf_surface;
        const unsigned int n_q_points_volume;
        const unsigned int n_q_points_surface;

        ConstraintMatrix constraints;
        SparsityPattern sparsity_pattern;

        SparseMatrix<double> tangent_matrix; /// system matrix
        Vector<double> system_rhs; /// system right hand side
        Vector<double> solution_n_volume; /// converged solution in volume

        Vector<double> solution_n_surface; /// converged solution restricted to surface

        Vector<double> X_volume; /// material placement of dof in volume
        Vector<double> x_volume; /// spatial placement of dof in volume

        Vector<double> X_surface; /// material placement of dof on surface
        Vector<double> x_surface; /// spatial placement of dof on surface

        /// map between cell on the surface and the cell it was extracted from in the volume
        std::map<typename DoFHandler<dim, spacedim>::cell_iterator,
            typename DoFHandler<spacedim, spacedim>::face_iterator> surface_to_volume_dof_iterator_map;

        /// map between dof on surface and in volume
        std::map<types::global_dof_index, types::global_dof_index> surface_to_volume_dof_map;

        /**
         * \brief Basic convergence error handling facility
         */
        struct Errors
        {
            Errors ()
                :
                    norm(1.)
            {
            }

            void
            reset ()
            {
              norm = 1.;
            }
            void
            normalise (
                const Errors & rhs)
            {
              if (rhs.norm != 0.)
                norm /= rhs.norm;
            }
            double norm;
        };

        Errors error_rhs, error_rhs_0, error_rhs_norm, error_delta_u,
            error_delta_u_0, error_delta_u_norm;

        /**
         * \brief Compute the error in the residual
         */
        void
        get_error_rhs ();

        /**
         * \brief Compute error associated with the iterative change in Newton scheme
         * @param newton_update Iterative change in solution
         */
        void
        get_error_delta_u (
            const Vector<double> & newton_update);

        /**
         * \brief Pretty print the convergence header
         */
        static
        void
        print_conv_header ();

        /**
         * \brief Pretty print the convergence footer
         */
        void
        print_conv_footer ();
    };

  template <int spacedim>
    Solid<spacedim>::Solid (
        const std::string & input_file)
        :
            parameters(input_file),
            vol_reference(0.0),
            vol_spatial(0.0),
            volume_triangulation(Triangulation<spacedim>::maximum_smoothing),
            the_time(parameters.end_time, parameters.pre_load_time,
                parameters.delta_t),
            timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
            poly_order(parameters.poly_degree),
            volume_mapping_ref(parameters.poly_degree),
            surface_mapping_ref(parameters.poly_degree),
            volume_fe(FE_Q<spacedim>(parameters.poly_degree), spacedim),
            surface_fe(FE_Q<dim, spacedim>(parameters.poly_degree), spacedim),
            volume_dof_handler(volume_triangulation),
            surface_dof_handler(surface_triangulation),
            dofs_per_volume_cell(volume_fe.dofs_per_cell),
            dofs_per_surface_cell(surface_fe.dofs_per_cell),
            u_fe(first_u_component),
            qf_volume(parameters.quad_order),
            qf_surface(parameters.quad_order),
            n_q_points_volume(qf_volume.size()),
            n_q_points_surface(qf_surface.size())
    {
    }

  template <int spacedim>
    Solid<spacedim>::~Solid ()
    {
      volume_dof_handler.clear();
      surface_dof_handler.clear();
    }

  template <int spacedim>
    void
    Solid<spacedim>::run ()
    {
      // make the grid
      make_grid();
      // initialise the sparsity pattern, system matrix and RHS. extract the surface grid.
      system_setup();
      setup_surface_volume_dof_map();
      X_surface = extract_surface_data(X_volume);
      x_surface = X_surface;
      // setup the data at the continuum points
      setup_cp_volume();
      setup_cp_surface();

      // output the data in the reference configuration
      output_results();
      the_time.increment();

      Vector<double> solution_delta; // change in solution during timestep
      solution_delta.reinit(volume_dof_handler.n_dofs());

      // loop over the time domain
      while (the_time.is_simulation_complete())
        {
          solution_delta = 0.0;
          // solve the incremental problem for the current timestep
          solve_nonlinear_timestep(solution_delta);
          // update the solution
          solution_n_volume += solution_delta;
          solution_n_surface += extract_surface_data(solution_delta);
          output_results();
          the_time.increment();
        }
    }

  /**
   * \brief Result of the computation of each volume cell's contribution to global system matrix and RHS
   */
  template <int spacedim>
    struct Solid<spacedim>::PerTaskData_Assemble_Volume
    {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;

        PerTaskData_Assemble_Volume (
            const unsigned int dofs_per_cell)
            :
                cell_matrix(dofs_per_cell, dofs_per_cell),
                cell_rhs(dofs_per_cell),
                local_dof_indices(dofs_per_cell)
        {
        }

        void
        reset ()
        {
          cell_matrix = 0.;
          cell_rhs = 0.;
        }
    };

  /**
   * \brief Temporary data structures needed for assembly on a volume cell
   */
  template <int spacedim>
    struct Solid<spacedim>::ScratchData_Assemble_Volume
    {
        FEValues<spacedim> fe_values_ref;
        FEFaceValues<spacedim> fe_face_values_ref;

        // shape function values and their gradients
        std::vector<std::vector<Tensor<1, spacedim> > > N;
        std::vector<std::vector<Tensor<2, spacedim> > > Grad_N;

        ScratchData_Assemble_Volume (
            const FiniteElement<spacedim> & fe_cell,
            const QGauss<spacedim> & qf_cell, const UpdateFlags uf_cell,
            const QGauss<spacedim - 1> & qf_face, const UpdateFlags uf_face)
            :
                fe_values_ref(fe_cell, qf_cell, uf_cell),
                fe_face_values_ref(fe_cell, qf_face, uf_face),
                N(qf_cell.size(),
                    std::vector<Tensor<1, spacedim> >(fe_cell.dofs_per_cell)),
                Grad_N(qf_cell.size(),
                    std::vector<Tensor<2, spacedim> >(fe_cell.dofs_per_cell))
        {
        }

        ScratchData_Assemble_Volume (
            const ScratchData_Assemble_Volume & rhs)
            :
                fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
                fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                    rhs.fe_face_values_ref.get_quadrature(),
                    rhs.fe_face_values_ref.get_update_flags()),
                N(rhs.N),
                Grad_N(rhs.Grad_N)
        {
        }

        void
        reset ()
        {
          const unsigned int n_q_points = Grad_N.size();
          const unsigned int n_dofs_per_cell = Grad_N[0].size();
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                N[q_point][k] = 0.;
                Grad_N[q_point][k] = 0.0;
              }
        }

    };

  /**
   * \brief Empty structure required by TBB
   */
  template <int spacedim>
    struct Solid<spacedim>::PerTaskData_UCP_Volume
    {
        void
        reset ()
        {
        }
    };

  /**
   * \brief Result of the computation of each surface cell's contribution to global system matrix and RHS
   */
  template <int spacedim>
    struct Solid<spacedim>::PerTaskData_Assemble_Surface
    {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        // the dof in the volume corresponding to those on the surface
        std::vector<types::global_dof_index> local_dof_indices_in_volume;

        PerTaskData_Assemble_Surface (
            const unsigned int dofs_per_cell)
            :
                cell_matrix(dofs_per_cell, dofs_per_cell),
                cell_rhs(dofs_per_cell),
                local_dof_indices(dofs_per_cell),
                local_dof_indices_in_volume(dofs_per_cell)
        {
        }

        void
        reset ()
        {
          cell_matrix = 0.;
          cell_rhs = 0.;
        }
    };

  /**
   * \brief Temporary data structures needed for assembly on a surface cell
   */
  template <int spacedim>
    struct Solid<spacedim>::ScratchData_Assemble_Surface
    {
        FEValues<dim, spacedim> fe_values_ref;
        // shape functions and their gradients
        std::vector<std::vector<Tensor<1, spacedim> > > N;
        std::vector<std::vector<Tensor<2, spacedim> > > Grad_N;

        ScratchData_Assemble_Surface (
            const FiniteElement<dim, spacedim> & fe_cell,
            const QGauss<dim> & qf_cell, const UpdateFlags uf_cell)
            :
                fe_values_ref(fe_cell, qf_cell, uf_cell),
                N(qf_cell.size(),
                    std::vector<Tensor<1, spacedim> >(fe_cell.dofs_per_cell)),
                Grad_N(qf_cell.size(),
                    std::vector<Tensor<2, spacedim> >(fe_cell.dofs_per_cell))
        {
        }

        ScratchData_Assemble_Surface (
            const ScratchData_Assemble_Surface & rhs)
            :
                fe_values_ref(rhs.fe_values_ref.get_fe(),
                    rhs.fe_values_ref.get_quadrature(),
                    rhs.fe_values_ref.get_update_flags()),
                N(rhs.N),
                Grad_N(rhs.Grad_N)
        {
        }

        void
        reset ()
        {
          const unsigned int n_q_points = Grad_N.size();
          const unsigned int n_dofs_per_cell = Grad_N[0].size();
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                N[q_point][k] = 0.;
                Grad_N[q_point][k] = 0.0;
              }
        }
    };

  /**
   * \brief Empty structure required by TBB
   */
  template <int spacedim>
    struct Solid<spacedim>::PerTaskData_UCP_Surface
    {
        void
        reset ()
        {
        }
    };

  /**
   * \brief Temporary data structures needed to update the CP data in the volume
   */
  template <int spacedim>
    struct Solid<spacedim>::ScratchData_UCP_Volume
    {

        // deformation gradient and its inverse at the quadrature points
        std::vector<Tensor<2, spacedim> > F_at_qps;
        std::vector<Tensor<2, spacedim> > F_inv_at_qps;

        // Jacobian determinant and its inverse at the quadrature points
        std::vector<double> JxW_material;
        std::vector<double> JxW_spatial;

        FEValues<spacedim> fe_values_spatial;
        FEValues<spacedim> fe_values_material;

        ScratchData_UCP_Volume (
            const FiniteElement<spacedim> & fe_cell,
            const DoFHandler<spacedim>& dof_cell,
            const QGauss<spacedim> & qf_cell, const UpdateFlags uf_cell,
            const MappingQEulerian<spacedim, Vector<double> > & q_eulerian)
            :
                F_at_qps(qf_cell.size()),
                F_inv_at_qps(qf_cell.size()),
                JxW_material(qf_cell.size()),
                JxW_spatial(qf_cell.size()),
                fe_values_spatial(q_eulerian, fe_cell, qf_cell, uf_cell),
                fe_values_material(fe_cell, qf_cell, uf_cell)
        {
        }

        ScratchData_UCP_Volume (
            const ScratchData_UCP_Volume & rhs)
            :
                F_at_qps(rhs.F_at_qps),
                F_inv_at_qps(rhs.F_inv_at_qps),
                JxW_material(rhs.JxW_material),
                JxW_spatial(rhs.JxW_spatial),
                fe_values_spatial(rhs.fe_values_spatial.get_mapping(),
                    rhs.fe_values_spatial.get_fe(),
                    rhs.fe_values_spatial.get_quadrature(),
                    rhs.fe_values_spatial.get_update_flags()),
                fe_values_material(rhs.fe_values_material.get_fe(),
                    rhs.fe_values_material.get_quadrature(),
                    rhs.fe_values_material.get_update_flags())
        {
        }

        void
        reset ()
        {
          const unsigned int n_q_points = F_at_qps.size();
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              F_at_qps[q] = 0.;
              F_inv_at_qps[q] = 0.;
              JxW_material[q] = 0.;
              JxW_spatial[q] = 0.;
            }
        }
    };

  /**
   * \brief Temporary data structures needed to update the CP data on the surface
   */
  template <int spacedim>
    struct Solid<spacedim>::ScratchData_UCP_Surface
    {

        // deformation gradient and its inverse at the quadrature points
        std::vector<Tensor<2, spacedim> > F_at_qps;
        std::vector<Tensor<2, spacedim> > F_inv_at_qps;
        // spatial normal evaluated at the quadrature points
        std::vector<Point<spacedim> > normals_at_qps;
        // surface identity tensor at the quadrature points
        std::vector<Tensor<2, spacedim> > identity_at_qps;

        // Jacobian determinant and its inverse at the quadrature points
        std::vector<double> JxW_material;
        std::vector<double> JxW_spatial;

        FEValues<dim, spacedim> fe_values_spatial;
        FEValues<dim, spacedim> fe_values_material;

        ScratchData_UCP_Surface (
            const FiniteElement<dim, spacedim> & fe_cell,
            const DoFHandler<dim, spacedim>& dof_cell,
            const QGauss<dim> & qf_cell, const UpdateFlags uf_cell,
            const MappingQEulerian<dim, Vector<double>, spacedim> & q_eulerian)
            :
                F_at_qps(qf_cell.size()),
                F_inv_at_qps(qf_cell.size()),
                normals_at_qps(qf_cell.size()),
                identity_at_qps(qf_cell.size()),
                JxW_material(qf_cell.size()),
                JxW_spatial(qf_cell.size()),
                fe_values_spatial(q_eulerian, fe_cell, qf_cell, uf_cell),
                fe_values_material(fe_cell, qf_cell, uf_cell)
        {
        }

        ScratchData_UCP_Surface (
            const ScratchData_UCP_Surface & rhs)
            :
                F_at_qps(rhs.F_at_qps),
                F_inv_at_qps(rhs.F_inv_at_qps),
                normals_at_qps(rhs.normals_at_qps),
                identity_at_qps(rhs.identity_at_qps),
                JxW_material(rhs.JxW_material),
                JxW_spatial(rhs.JxW_spatial),
                fe_values_spatial(rhs.fe_values_spatial.get_mapping(),
                    rhs.fe_values_spatial.get_fe(),
                    rhs.fe_values_spatial.get_quadrature(),
                    rhs.fe_values_spatial.get_update_flags()),
                fe_values_material(rhs.fe_values_material.get_fe(),
                    rhs.fe_values_material.get_quadrature(),
                    rhs.fe_values_material.get_update_flags())
        {
        }

        void
        reset ()
        {
          const unsigned int n_q_points = F_at_qps.size();
          for (unsigned int qp = 0; qp < n_q_points; ++qp)
            {
              F_at_qps[qp] = 0.;
              F_inv_at_qps[qp] = 0.;
              normals_at_qps[qp] = Point<spacedim>();
              identity_at_qps[qp] = 0.;
              JxW_material[qp] = 0.;
              JxW_spatial[qp] = 0.;
            }
        }
    };

  template <int spacedim>
    void
    Solid<spacedim>::set_data_beginning_time_step ()
    {

      const double ramp_factor = the_time.get_pre_load_factor();

      // volume
      for (typename DoFHandler<spacedim>::active_cell_iterator cell =
          volume_dof_handler.begin_active(); cell != volume_dof_handler.end();
          ++cell)
        {
          std::vector<ContinuumPoint<spacedim> > &lqph = volume_cp_map[cell];

          for (unsigned int q_count = 0; q_count < n_q_points_volume; q_count++)
            lqph[q_count].set_data_each_time_step(ramp_factor);

        }

      // surface
      for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
          surface_triangulation.begin_active();
          cell != surface_triangulation.end(); ++cell)
        {
          std::vector<ContinuumPoint<spacedim> > &lqph = surface_cp_map[cell];

          for (unsigned int q_count = 0; q_count < n_q_points_surface;
              q_count++)
            lqph[q_count].set_data_each_time_step(ramp_factor);

        }

    }

  template <int spacedim>
    void
    Solid<spacedim>::make_grid ()
    {
      timer.enter_subsection("Construct grid");
      std::cout << "    Making the grid..." << std::endl;
      GridIn<spacedim> bulk_grid_in;
      bulk_grid_in.attach_triangulation(volume_triangulation);

      std::string input_deck = parameters.input_deck;
      std::ifstream input_stream(input_deck.c_str());
      bulk_grid_in.read_ucd(input_stream);

      volume_triangulation.refine_global(parameters.global_refinement);

      vol_reference = GridTools::volume(volume_triangulation);
      vol_spatial = vol_reference;
      std::cout << "Grid:\n\t Material volume: " << vol_reference << std::endl;
      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::system_setup ()
    {
      timer.enter_subsection("Setup system");

      volume_dof_handler.distribute_dofs(volume_fe);
      DoFRenumbering::Cuthill_McKee(volume_dof_handler);

      // boundary id of the energetic surface(s)
      std::set<types::boundary_id> boundary_ids;
      if (parameters.problem_description == "nanowire"
          || parameters.problem_description == "bridge")
        boundary_ids.insert(1);
      else if (parameters.problem_description == "rough_surface")
        boundary_ids.insert(6);
      else if (parameters.problem_description == "cooks")
        {
          boundary_ids.insert(3);
          boundary_ids.insert(4);
          boundary_ids.insert(5);
          boundary_ids.insert(6);
        }

      // create the surface mesh from the boundary of the volume.
      // an exception will be thrown of the boundary id does not exist.
      surface_to_volume_dof_iterator_map = GridTools::extract_boundary_mesh(
          volume_dof_handler, surface_dof_handler, boundary_ids);

      surface_dof_handler.distribute_dofs(surface_fe);

      std::cout << "Volume triangulation:" << "\n\t Number of active cells: "
          << volume_triangulation.n_active_cells()
          << "\n\t Number of degrees of freedom: "
          << volume_dof_handler.n_dofs() << std::endl;

      std::cout << "Surface triangulation:" << "\n\t Number of active cells: "
          << surface_triangulation.n_active_cells()
          << "\n\t Number of degrees of freedom: "
          << surface_dof_handler.n_dofs() << std::endl;

      tangent_matrix.clear();
        {
          CompressedSimpleSparsityPattern csp(volume_dof_handler.n_dofs());
          DoFTools::make_sparsity_pattern(volume_dof_handler, csp, constraints,
              false);
          sparsity_pattern.copy_from(csp);
        }
      tangent_matrix.reinit(sparsity_pattern);
      system_rhs.reinit(volume_dof_handler.n_dofs());
      solution_n_volume.reinit(volume_dof_handler.n_dofs());
      X_volume.reinit(volume_dof_handler.n_dofs());
      x_volume.reinit(volume_dof_handler.n_dofs());

      solution_n_surface.reinit(surface_dof_handler.n_dofs());
      X_surface.reinit(surface_dof_handler.n_dofs());
      x_surface.reinit(surface_dof_handler.n_dofs());

      // populate the vectors of the material and spatial coordinates
      std::vector<Point<spacedim> > support_points(volume_dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(volume_mapping_ref,
          volume_dof_handler, support_points);

      FEValuesExtractors::Scalar x_extract(0);
      FEValuesExtractors::Scalar y_extract(1);
      FEValuesExtractors::Scalar z_extract(2);

      ComponentMask x_mask = volume_fe.component_mask(x_extract);
      ComponentMask y_mask = volume_fe.component_mask(y_extract);
      ComponentMask z_mask = volume_fe.component_mask(z_extract);

      std::vector<bool> x_dof(volume_dof_handler.n_dofs());
      std::vector<bool> y_dof(volume_dof_handler.n_dofs());
      std::vector<bool> z_dof(volume_dof_handler.n_dofs());

      DoFTools::extract_dofs(volume_dof_handler, x_mask, x_dof);
      DoFTools::extract_dofs(volume_dof_handler, y_mask, y_dof);
      DoFTools::extract_dofs(volume_dof_handler, z_mask, z_dof);

      // populate X and x_volume
      for (types::global_dof_index ii = 0; ii < volume_dof_handler.n_dofs();
          ii++)
        {
          Point<spacedim> dof_support_point = support_points[ii];
          unsigned int component = spacedim;
          if (x_dof[ii])
            component = 0;
          else if (y_dof[ii])
            component = 1;
          else if (z_dof[ii])
            component = 2;

          X_volume[ii] = dof_support_point[component];
        }
      x_volume = X_volume;

      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::setup_cp_volume ()
    {
      std::cout << "    Setting up continuum point data map in volume..."
          << std::endl;

      const ContinuumPoint<spacedim> volume_continuum_point_temp(0);

      std::vector<ContinuumPoint<spacedim> > cell_point_history(
          n_q_points_volume, volume_continuum_point_temp);

      FEValues<spacedim> fe_values(volume_fe, qf_volume,
          update_values | update_quadrature_points);

      for (typename Triangulation<spacedim>::active_cell_iterator cell =
          volume_triangulation.begin_active();
          cell != volume_triangulation.end(); ++cell)
        {
          fe_values.reinit(cell);
          volume_cp_map[cell] = cell_point_history;

          for (unsigned int q_point = 0; q_point < n_q_points_volume; ++q_point)
            {
              Point<spacedim> pos_ref_qp = fe_values.quadrature_point(q_point);

              (volume_cp_map[cell][q_point]).initialise_data(pos_ref_qp,
                  parameters.lambda_volume, parameters.mu_volume);
            }
        }
    }

  template <int spacedim>
    void
    Solid<spacedim>::setup_cp_surface ()
    {
      std::cout << "    Setting up continuum point data map on surface..."
          << std::endl;

      // need to compute the load factor "manually" as the Time class has not yet been setup
      const double load_factor = parameters.delta_t / parameters.pre_load_time;

      const ContinuumPoint<spacedim> surface_continuum_point_temp(1);
      std::vector<ContinuumPoint<spacedim> > cell_point_history(
          n_q_points_surface, surface_continuum_point_temp);

      // vector containing the material surface identity
      std::vector<Tensor<2, spacedim> > I_hat_at_qps(n_q_points_surface);

      FEValues<dim, spacedim> fe_values(surface_fe, qf_surface,
          update_values | update_gradients | update_normal_vectors
              | update_quadrature_points);

      for (typename DoFHandler<dim, spacedim>::active_cell_iterator cell =
          surface_dof_handler.begin_active(), endc = surface_dof_handler.end();
          cell != endc; ++cell)
        {

          fe_values.reinit(cell);
          surface_cp_map[cell] = cell_point_history;
          // compute as the material surface gradient of material points
          fe_values[u_fe].get_function_gradients(X_surface, I_hat_at_qps);

          for (unsigned int q_point = 0; q_point < n_q_points_surface;
              ++q_point)
            {
              const Point<spacedim> position = fe_values.quadrature_point(
                  q_point);
              const Tensor<1, spacedim> normal = fe_values.normal_vector(
                  q_point);
              const Tensor<2, spacedim> I_hat = I_hat_at_qps[q_point];

              (surface_cp_map[cell][q_point]).initialise_data(position,
                  parameters.lambda_surface, parameters.mu_surface,
                  parameters.gamma_surface, load_factor, I_hat, normal);
            }
        }
    }

  template <int spacedim>
    void
    Solid<spacedim>::update_volume_cp_incremental (
        const Vector<double> & solution_delta)
    {
      timer.enter_subsection("Update CP data");
      std::cout << " UCP_v " << std::flush;

      const Vector<double> solution_total(
          get_total_solution_volume(solution_delta));

      const UpdateFlags uf_UQPH(
          update_values | update_gradients | update_JxW_values);

      // spatial mapping
      const MappingQEulerian<spacedim, Vector<double> > q_mapping(poly_order,
          solution_total, volume_dof_handler);

      PerTaskData_UCP_Volume per_task_data_UQPH;
      ScratchData_UCP_Volume scratch_data_UQPH(volume_fe, volume_dof_handler,
          qf_volume, uf_UQPH, q_mapping);

      WorkStream::run(volume_dof_handler.begin_active(),
          volume_dof_handler.end(), *this,
          &Solid::update_volume_cp_incremental_one_cell,
          &Solid::copy_local_to_global_ucp_volume, scratch_data_UQPH,
          per_task_data_UQPH);

      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::update_surface_cp_incremental (
        const Vector<double> & solution_delta_volume)
    {
      timer.enter_subsection("Update CP data");
      std::cout << " UCP_s " << std::flush;

      const Vector<double> solution_total_surface = get_total_solution_surface(
          solution_delta_volume);

      const UpdateFlags uf_UQPH(
          update_values | update_gradients | update_normal_vectors
              | update_quadrature_points | update_JxW_values);

      // spatial mapping
      const MappingQEulerian<dim, Vector<double>, spacedim> q_mapping(
          poly_order, solution_total_surface, surface_dof_handler);

      PerTaskData_UCP_Surface per_task_data_UQPH;
      ScratchData_UCP_Surface scratch_data_UQPH(surface_fe, surface_dof_handler,
          qf_surface, uf_UQPH, q_mapping);

      WorkStream::run(surface_dof_handler.begin_active(),
          surface_dof_handler.end(), *this,
          &Solid::update_surface_cp_incremental_one_cell,
          &Solid::copy_local_to_global_ucp_surface, scratch_data_UQPH,
          per_task_data_UQPH);

      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::update_volume_cp_incremental_one_cell (
        const typename DoFHandler<spacedim>::active_cell_iterator & cell,
        ScratchData_UCP_Volume & scratch, PerTaskData_UCP_Volume & data)
    {

      std::vector<ContinuumPoint<spacedim> >& lqph = volume_cp_map[cell];

      scratch.reset();

      // create both a material and spatial view of the cell
      scratch.fe_values_material.reinit(cell);
      scratch.fe_values_spatial.reinit(cell);

      // compute the deformation gradient as material gradient of spatial placement
      scratch.fe_values_material[u_fe].get_function_gradients(x_volume,
          scratch.F_at_qps);

      // compute the inverse deformation gradient as spatial gradient of material placement
      scratch.fe_values_spatial[u_fe].get_function_gradients(X_volume,
          scratch.F_inv_at_qps);

      // the determinant of the mapping from the material to the spatial
      // configuration is computed as the ratio of the mapping from the isoparametric
      // to the spatial divided by the ratio of the mapping from the isoparametric
      // to the material (the quadrature weighting is the same)
      scratch.JxW_material = scratch.fe_values_material.get_JxW_values();
      scratch.JxW_spatial = scratch.fe_values_spatial.get_JxW_values();

      for (unsigned int q_point = 0; q_point < n_q_points_volume; ++q_point)
        lqph[q_point].update_data(scratch.F_at_qps[q_point],
            scratch.F_inv_at_qps[q_point],
            scratch.JxW_spatial[q_point] / scratch.JxW_material[q_point]);
    }

  template <int spacedim>
    void
    Solid<spacedim>::update_surface_cp_incremental_one_cell (
        const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
        ScratchData_UCP_Surface & scratch, PerTaskData_UCP_Surface & data)
    {

      std::vector<ContinuumPoint<spacedim> >& lqph = surface_cp_map[cell];

      scratch.reset();

      // create both a material and spatial view of the cell
      scratch.fe_values_material.reinit(cell);
      scratch.fe_values_spatial.reinit(cell);

      // compute the (rank deficient)  deformation gradient
      // as material surface gradient of spatial placement
      scratch.fe_values_material[u_fe].get_function_gradients(x_surface,
          scratch.F_at_qps);

      // compute the inverse of the rank-deficient deformation gradient
      // as spatial surface gradient of material placement,
      scratch.fe_values_spatial[u_fe].get_function_gradients(X_surface,
          scratch.F_inv_at_qps);

      scratch.fe_values_spatial[u_fe].get_function_gradients(x_surface,
          scratch.identity_at_qps);

      scratch.normals_at_qps = scratch.fe_values_spatial.get_normal_vectors();

      scratch.JxW_material = scratch.fe_values_material.get_JxW_values();
      scratch.JxW_spatial = scratch.fe_values_spatial.get_JxW_values();

      for (unsigned int q = 0; q < n_q_points_surface; ++q)
        lqph[q].update_data(scratch.F_at_qps[q], scratch.F_inv_at_qps[q],
            scratch.JxW_spatial[q] / scratch.JxW_material[q],
            scratch.identity_at_qps[q], scratch.normals_at_qps[q]);
    }

  template <int spacedim>
    void
    Solid<spacedim>::solve_nonlinear_timestep (
        Vector<double> & solution_delta)
    {
      std::cout << std::endl << "Timestep " << the_time.get_timestep() << " @ "
          << the_time.current() << "s" << std::endl;

      Vector<double> newton_update;
      newton_update.reinit(volume_dof_handler.n_dofs());

      // reset all of the error measures used in the Newton scheme
      error_rhs.reset();
      error_rhs_0.reset();
      error_rhs_norm.reset();
      error_delta_u.reset();
      error_delta_u_0.reset();
      error_delta_u_norm.reset();

      print_conv_header();

      // solve the problem for each Newton iteration
      unsigned int newton_iteration = 0;
      for (; newton_iteration < parameters.max_iterations_NR;
          ++newton_iteration)
        {

          std::cout << " " << std::setw(2) << newton_iteration << " "
              << std::flush;

          // set initial data for the time step
          if (newton_iteration == 0)
            set_data_beginning_time_step();

          // reset the system matrix and the RHS
          tangent_matrix = 0.;
          system_rhs = 0.;

          // create the constraints
          make_constraints(newton_iteration);

          // assemble the system matrix and RHS
          assemble_system_volume();
          assemble_system_surface();

          // solve the linear problem
          const std::pair<unsigned int, double> lin_solver_output =
              solve_linear_system(newton_update);

          // update the solution and the spatial coordinates
          solution_delta += newton_update;
          x_volume += newton_update;
          x_surface = extract_surface_data(x_volume);

          // update the continuum point data
          update_volume_cp_incremental(solution_delta);
          update_surface_cp_incremental(solution_delta);

          // distribute the constraints to the RHS
          constraints.distribute(system_rhs);

          // compute the norm of the right-hand side
          get_error_rhs();
          error_rhs_norm = error_rhs;
          get_error_delta_u(newton_update);
          error_delta_u_norm = error_delta_u;

          // record the error at the zeroth iteration in order to normalise the residuals
          if (newton_iteration == 0)
            {
              error_delta_u_0 = error_delta_u;
              error_rhs_0 = error_rhs;
            }
          error_delta_u_norm.normalise(error_delta_u_0);
          error_rhs_norm.normalise(error_rhs_0);

          std::cout << " | " << std::fixed << std::setprecision(3)
              << std::setw(7) << std::scientific << lin_solver_output.first
              << "  " << lin_solver_output.second << "  " << error_rhs_norm.norm
              << "  " << error_delta_u_norm.norm << std::endl;

          // check the value of the error and see if problem has converged
          if (newton_iteration > 1
              && error_delta_u_norm.norm <= parameters.tol_soln
              && error_rhs_norm.norm <= parameters.tol_residual)
            {
              std::cout << " CONVERGED! " << std::endl;
              print_conv_footer();
              break;
            }
        }

      AssertThrow(newton_iteration < parameters.max_iterations_NR,
          ExcMessage("No convergence in nonlinear solver!"));
    }

  template <int spacedim>
    void
    Solid<spacedim>::print_conv_header ()
    {
      static const unsigned int l_width = 85;

      for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
      std::cout << std::endl;

      std::cout << "               SOLVER STEP                "
          << " |  LIN_IT  LIN_RES   |" << " |R_NORM|" << "  |dU_NORM|"

          << std::endl;

      for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
      std::cout << std::endl;
    }

  template <int spacedim>
    void
    Solid<spacedim>::print_conv_footer ()
    {
      static const unsigned int l_width = 85;

      for (unsigned int i = 0; i < l_width; ++i)
        std::cout << "_";
      std::cout << std::endl;

      std::cout << "Relative errors:" << std::endl << "\tSolution: |dU_NORM|\t"
          << error_delta_u_norm.norm << std::endl << "\tResidual: |R_NORM|\t"
          << error_rhs_norm.norm << std::endl;
    }

  template <int spacedim>
    void
    Solid<spacedim>::get_error_rhs ()
    {
      Vector<double> error_rhs_vec(volume_dof_handler.n_dofs());

      for (unsigned int i = 0; i < volume_dof_handler.n_dofs(); ++i)
        if (!constraints.is_constrained(i))
          error_rhs_vec(i) = system_rhs(i);

      error_rhs.norm = error_rhs_vec.l2_norm();
    }

  template <int spacedim>
    void
    Solid<spacedim>::get_error_delta_u (
        const Vector<double> & newton_update)
    {
      Vector<double> error_delta_u_vec(volume_dof_handler.n_dofs());

      for (unsigned int i = 0; i < volume_dof_handler.n_dofs(); ++i)
        if (!constraints.is_constrained(i))
          error_delta_u_vec(i) = newton_update(i);

      error_delta_u.norm = error_delta_u_vec.l2_norm();
    }

  template <int spacedim>
    Vector<double>
    Solid<spacedim>::get_total_solution_volume (
        const Vector<double> & solution_delta) const
    {
      Vector<double> solution_total(solution_n_volume);
      solution_total += solution_delta;
      return solution_total;
    }

  template <int spacedim>
    Vector<double>
    Solid<spacedim>::get_total_solution_surface (
        const Vector<double> & solution_delta)
    {
      Vector<double> solution_total(solution_n_surface);
      Vector<double> solution_delta_surface = extract_surface_data(
          solution_delta);
      solution_total += solution_delta_surface;
      return solution_total;
    }

  template <int spacedim>
    Vector<double>
    Solid<spacedim>::extract_surface_data (
        const Vector<double>& volume_data)
    {

      Vector<double> surface_data(surface_dof_handler.n_dofs());

      for (types::global_dof_index ii = 0; ii < surface_dof_handler.n_dofs();
          ii++)
        {
          types::global_dof_index volume_index = surface_to_volume_dof_map[ii];
          surface_data[ii] = volume_data[volume_index];

        }
      return surface_data;
    }

  template <int spacedim>
    void
    Solid<spacedim>::assemble_system_volume ()
    {
      timer.enter_subsection("Assemble system volume");
      std::cout << " ASS_v " << std::flush;

      tangent_matrix = 0.;

      const UpdateFlags uf_cell(
          update_values | update_gradients | update_JxW_values);
      const UpdateFlags uf_face(
          update_values | update_normal_vectors | update_JxW_values);

      PerTaskData_Assemble_Volume per_task_data(dofs_per_volume_cell);
      ScratchData_Assemble_Volume scratch_data(volume_fe, qf_volume, uf_cell,
          qf_surface, uf_face);

      WorkStream::run(volume_dof_handler.begin_active(),
          volume_dof_handler.end(), *this,
          &Solid::assemble_system_one_cell_volume,
          &Solid::copy_local_to_global_volume, scratch_data, per_task_data);

      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::assemble_system_one_cell_volume (
        const typename DoFHandler<spacedim>::active_cell_iterator & cell,
        ScratchData_Assemble_Volume & scratch,
        PerTaskData_Assemble_Volume & data)
    {
      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      std::vector<ContinuumPoint<spacedim> > & lqph = volume_cp_map[cell];

      // precompute Grad of the interpolation functions at the qps
      for (unsigned int q = 0; q < n_q_points_volume; ++q)
        for (unsigned int k = 0; k < dofs_per_volume_cell; ++k)
          {
            scratch.N[q][k] = scratch.fe_values_ref[u_fe].value(k, q);
            scratch.Grad_N[q][k] = scratch.fe_values_ref[u_fe].gradient(k, q);
          }

      for (unsigned int q = 0; q < n_q_points_volume; ++q)
        {

          // the stress
          const Tensor<2, spacedim> P = lqph[q].get_P();
          // the material tangent
          const Tensor<4, spacedim> AA = lqph[q].get_AA();

          const std::vector<Tensor<2, spacedim> > & Grad_N = scratch.Grad_N[q];

          const double J_iso_xW = scratch.fe_values_ref.JxW(q);

          for (unsigned int i = 0; i < dofs_per_volume_cell; ++i)
            {
              // compute the residual contribution to the RHS
              data.cell_rhs(i) -= scalar_product(Grad_N[i], P) * J_iso_xW;
              // compute the contribution to the tangent
              for (unsigned int j = 0; j < dofs_per_volume_cell; ++j)
                {
                  Tensor<2, spacedim> A_ddot_Grad_N_j;
                  A_ddot_Grad_N_j = AdditionalTools::contract_ijkl_kl(AA,
                      Grad_N[j]);
                  data.cell_matrix(i, j) += scalar_product(Grad_N[i],
                      A_ddot_Grad_N_j) * J_iso_xW;
                }
            }
        }

      // assemble Neumann terms if specified (done for the volume only)
      if (parameters.neumann_surface_id != -1)
        {
          const double load_factor = the_time.get_pre_load_factor();
          for (unsigned int face = 0;
              face < GeometryInfo<spacedim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary() == true
                && cell->face(face)->boundary_indicator()
                    == parameters.neumann_surface_id)
              {
                scratch.fe_face_values_ref.reinit(cell, face);

                // prescribed Neumann traction vector
                Tensor<1, spacedim> traction;

                traction[0] = load_factor * parameters.traction_x;
                traction[1] = load_factor * parameters.traction_y;
                traction[2] = load_factor * parameters.traction_z;

                for (unsigned int q_point = 0; q_point < n_q_points_surface;
                    ++q_point)
                  {
                    // J_xW for the boundary of the volume
                    const double J_iso_xW_surf = scratch.fe_face_values_ref.JxW(
                        q_point);
                    // loop over all the dofs in the volume (NB)
                    for (unsigned int i = 0; i < dofs_per_volume_cell; ++i)
                      {
                        // non-zero component of the shape function
                        const unsigned int component_i =
                            volume_fe.system_to_component_index(i).first;
                        // value of the shape function on the boundary of the cell
                        const double N_i =
                            scratch.fe_face_values_ref.shape_value(i, q_point);
                        data.cell_rhs(i) += N_i * traction[component_i]
                            * J_iso_xW_surf;
                      }
                  }
              }

        }
    }

  template <int spacedim>
    void
    Solid<spacedim>::copy_local_to_global_volume (
        const PerTaskData_Assemble_Volume & data)
    {
      // constraints are imposed as the element stiffness matrix is added to the global
      constraints.distribute_local_to_global(data.cell_matrix, data.cell_rhs,
          data.local_dof_indices, tangent_matrix, system_rhs);
    }

  template <int spacedim>
    void
    Solid<spacedim>::assemble_system_surface ()
    {
      timer.enter_subsection("Assemble system surface");
      std::cout << " ASS_s " << std::flush;

      const UpdateFlags uf_cell(
          update_values | update_gradients | update_JxW_values);

      PerTaskData_Assemble_Surface per_task_data(dofs_per_surface_cell);
      ScratchData_Assemble_Surface scratch_data(surface_fe, qf_surface,
          uf_cell);

      WorkStream::run(surface_dof_handler.begin_active(),
          surface_dof_handler.end(), *this,
          &Solid::assemble_system_one_cell_surface,
          &Solid::copy_local_to_global_surface, scratch_data, per_task_data);

      timer.leave_subsection();
    }

  template <int spacedim>
    void
    Solid<spacedim>::assemble_system_one_cell_surface (
        const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
        ScratchData_Assemble_Surface & scratch,
        PerTaskData_Assemble_Surface & data)
    {
      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      std::vector<ContinuumPoint<spacedim> > &lqph = surface_cp_map[cell];

      // compute the corresponding local dof indices on the material boundary of the volume
      for (unsigned int k = 0; k < dofs_per_surface_cell; ++k)
        {
          data.local_dof_indices_in_volume[k] =
              surface_to_volume_dof_map[data.local_dof_indices[k]];

        }

      for (unsigned int q_point = 0; q_point < n_q_points_surface; ++q_point)
        {
          for (unsigned int k = 0; k < dofs_per_surface_cell; ++k)
            {
              scratch.N[q_point][k] = scratch.fe_values_ref[u_fe].value(k,
                  q_point);
              scratch.Grad_N[q_point][k] = scratch.fe_values_ref[u_fe].gradient(
                  k, q_point);
            }
        }

      // this follows in an identical fashion to the volume
      for (unsigned int q_point = 0; q_point < n_q_points_surface; ++q_point)
        {

          const Tensor<2, spacedim> P = lqph[q_point].get_P();
          const Tensor<4, spacedim> AA = lqph[q_point].get_AA();

          const std::vector<Tensor<2, spacedim> > & Grad_N =
              scratch.Grad_N[q_point];

          const double J_iso_xW = scratch.fe_values_ref.JxW(q_point);

          for (unsigned int i = 0; i < dofs_per_surface_cell; ++i)
            {
              data.cell_rhs(i) -= scalar_product(Grad_N[i], P) * J_iso_xW;
              for (unsigned int j = 0; j < dofs_per_surface_cell; ++j)
                {
                  Tensor<2, spacedim> A_ddot_Grad_N_j;
                  A_ddot_Grad_N_j = AdditionalTools::contract_ijkl_kl(AA,
                      Grad_N[j]);

                  data.cell_matrix(i, j) += scalar_product(Grad_N[i],
                      A_ddot_Grad_N_j) * J_iso_xW;
                }
            }
        }
    }

  template <int spacedim>
    void
    Solid<spacedim>::copy_local_to_global_surface (
        const PerTaskData_Assemble_Surface & data)
    {
      // copy the element stiffness matrix on the surface to the
      // global system matrix of the volume by using the map between
      // the dof

      constraints.distribute_local_to_global(data.cell_matrix, data.cell_rhs,
          data.local_dof_indices_in_volume, tangent_matrix, system_rhs);

    }

  template <int spacedim>
    void
    Solid<spacedim>::make_constraints (
        const int & it_nr)
    {
      std::cout << " CST " << std::flush;

      const double disp = the_time.get_delta_t()
          * parameters.final_displacement;

      // only impose homogeneous constraints once per timestep
      if (it_nr > 1)
        return;

      constraints.clear();
      const bool apply_dirichlet_bc = (it_nr == 0);

      if (parameters.problem_description == "nanowire")
        {

            { // x component; -z face
              const types::boundary_id boundary_id = 2;
              const unsigned int comp_to_set = 0;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

            { // y component; -z face
              const types::boundary_id boundary_id = 2;
              const unsigned int comp_to_set = 1;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

            { // z component; -z face
              const types::boundary_id boundary_id = 2;
              const unsigned int comp_to_set = 2;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              if (apply_dirichlet_bc == true)
                VectorTools::interpolate_boundary_values(volume_dof_handler,
                    boundary_id, ConstantFunction<spacedim>(-disp, spacedim),
                    constraints, components);
              else
                VectorTools::interpolate_boundary_values(volume_dof_handler,
                    boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                    components);
            }

            { // x component; +z face
              const types::boundary_id boundary_id = 3;
              const unsigned int comp_to_set = 0;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

            { // y component; +z face
              const types::boundary_id boundary_id = 3;
              const unsigned int comp_to_set = 1;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

            { // z component; +z face
              const types::boundary_id boundary_id = 3;
              const unsigned int comp_to_set = 2;
              std::vector<bool> components(spacedim, false);
              components[comp_to_set] = true;

              if (apply_dirichlet_bc == true)
                VectorTools::interpolate_boundary_values(volume_dof_handler,
                    boundary_id, ConstantFunction<spacedim>(disp, spacedim),
                    constraints, components);
              else
                VectorTools::interpolate_boundary_values(volume_dof_handler,
                    boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                    components);
            }
        }
      else if (parameters.problem_description == "bridge")
        {

            { // all components, -z face
              const types::boundary_id boundary_id = 2;
              std::vector<bool> components(spacedim, true);

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

            { // all components, z face
              const types::boundary_id boundary_id = 3;
              std::vector<bool> components(spacedim, true);

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

        }
      else if (parameters.problem_description == "rough_surface")
        {
            { // all components, -x face
              const types::boundary_id boundary_id = 1;
              std::vector<bool> components(spacedim, true);

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }
        }
      else if (parameters.problem_description == "cooks")
        {
            { // all components, -x face
              const types::boundary_id boundary_id = 1;
              std::vector<bool> components(spacedim, true);

              VectorTools::interpolate_boundary_values(volume_dof_handler,
                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
                  components);
            }

//            { // z component, -z face
//              const types::boundary_id boundary_id = 5;
//              const unsigned int comp_to_set = 2;
//              std::vector<bool> components(spacedim, false);
//              components[comp_to_set] = true;
//
//              VectorTools::interpolate_boundary_values(volume_dof_handler,
//                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
//                  components);
//            }
//
//            { // z component, +z face
//              const types::boundary_id boundary_id = 6;
//              const unsigned int comp_to_set = 2;
//              std::vector<bool> components(spacedim, false);
//              components[comp_to_set] = true;
//
//              VectorTools::interpolate_boundary_values(volume_dof_handler,
//                  boundary_id, ZeroFunction<spacedim>(spacedim), constraints,
//                  components);
//            }

        }

      constraints.close();
    }

  template <int spacedim>
    std::pair<unsigned int, double>
    Solid<spacedim>::solve_linear_system (
        Vector<double> & newton_update)
    {

      unsigned int lin_it = 0;
      double lin_res = 0.0;

      timer.enter_subsection("Linear solver");
      std::cout << " SLV " << std::flush;
      if (parameters.type_lin == "CG")
        {
          const int solver_its = tangent_matrix.m()
              * parameters.max_iterations_lin;
          const double tol_sol = parameters.tol_lin * system_rhs.l2_norm();

          SolverControl solver_control(solver_its, tol_sol);
          GrowingVectorMemory<Vector<double> > GVM;
          SolverCG<Vector<double> > solver_CG(solver_control, GVM);

          PreconditionSelector<SparseMatrix<double>, Vector<double> > preconditioner(
              parameters.preconditioner_type,
              parameters.preconditioner_relaxation);
          preconditioner.use_matrix(tangent_matrix);
          solver_CG.solve(tangent_matrix, newton_update, system_rhs,
              preconditioner);
          lin_it = solver_control.last_step();
          lin_res = solver_control.last_value();
        }
      else if (parameters.type_lin == "Direct")
        {
          Vector<double> dx;
          Vector<double> b;
          dx.reinit(volume_dof_handler.n_dofs());
          b.reinit(volume_dof_handler.n_dofs());
          b = system_rhs;

          SparseDirectUMFPACK A;
          A.initialize(tangent_matrix);
          A.vmult(dx, b);
          newton_update = dx;

          lin_it = 1;
          lin_res = 0.;
        }
      else
        AssertThrow(false, ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();

      constraints.distribute(newton_update);

      return std::make_pair(lin_it, lin_res);
    }

  template <int spacedim>
    void
    Solid<spacedim>::output_results ()
    {
      timer.enter_subsection("Postprocess results");

      double psi_volume = 0.;
      double psi_surface = 0.;
        {
          // volume
          DataOut<spacedim> data_out;
          std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
              spacedim,
              DataComponentInterpretation::component_is_part_of_vector);

          std::vector<std::string> displacement(spacedim, "displacement");

          data_out.attach_dof_handler(volume_dof_handler);

          data_out.add_data_vector(solution_n_volume, displacement,
              DataOut<spacedim>::type_dof_data, data_component_interpretation);

          Vector<double> P_norm(volume_triangulation.n_active_cells()),
              sigma_norm(volume_triangulation.n_active_cells()), J(
                  volume_triangulation.n_active_cells());

          // data at the continuum points
          std::vector<Point<spacedim> > points; // position
          std::vector<double> J_qp; // J
          std::vector<Tensor<2, spacedim> > P_qp; // Piola-Kirchhoff stress
          std::vector<Tensor<2, spacedim> > sigma_qp; // Cauchy stress
          std::vector<Tensor<2, spacedim> > F_qp; // F

          FEValues<spacedim> fe_values(volume_fe, qf_volume,
              update_values | update_quadrature_points | update_JxW_values);

          typename DoFHandler<spacedim>::active_cell_iterator cell =
              volume_dof_handler.begin_active(), endc =
              volume_dof_handler.end();

          //  the P:F norm in the volume
          double P_F_norm_volume = 0.;

          for (unsigned int cell_count = 0; cell != endc; ++cell, ++cell_count)
            {
              fe_values.reinit(cell);
              Tensor<2, spacedim> P_avg; // average Piola-Kirchhoff stress
              double J_avg = 0.; // average J
              Tensor<2, spacedim> sigma_avg; // average Cauchy stress

              std::vector<ContinuumPoint<spacedim> > &cell_point_history_vol =
                  volume_cp_map[cell];

              for (unsigned int qq = 0; qq < n_q_points_volume; qq++)
                {
                  P_avg += cell_point_history_vol[qq].get_P();
                  sigma_avg += cell_point_history_vol[qq].get_sigma();
                  J_avg += cell_point_history_vol[qq].get_J();

                  J_qp.push_back(cell_point_history_vol[qq].get_J());
                  P_qp.push_back(cell_point_history_vol[qq].get_P());
                  sigma_qp.push_back(cell_point_history_vol[qq].get_sigma());
                  F_qp.push_back(cell_point_history_vol[qq].get_F());
                  Point<spacedim> temp_point = fe_values.quadrature_point(qq);
                  points.push_back(temp_point);

                  const double F_P_2_cp = std::pow(
                      double_contract(cell_point_history_vol[qq].get_F(),
                          cell_point_history_vol[qq].get_P()), 2);
                  P_F_norm_volume += F_P_2_cp * fe_values.JxW(qq);

                  psi_volume += cell_point_history_vol[qq].get_psi()
                      * fe_values.JxW(qq);
                }

              // average over the quadrature points
              P_avg *= (1.0 / n_q_points_volume);
              sigma_avg *= (1.0 / n_q_points_volume);
              J_avg *= (1.0 / n_q_points_volume);

              P_norm(cell_count) = P_avg.norm();

              sigma_norm(cell_count) = sigma_avg.norm();
              J(cell_count) = J_avg;
            }

          P_F_norm_volume = std::sqrt(P_F_norm_volume);

//          std::cout << "\tP:F norm in volume:\t" << std::fixed << std::setprecision(5) << std::scientific << P_F_norm_volume
//              << std::endl;
//          std::cout << "\tEnergy in volume:\t" << std::fixed << std::setprecision(5) << std::scientific << psi_volume
//                 << std::endl;

          data_out.add_data_vector(P_norm, "P_norm");
          data_out.add_data_vector(sigma_norm, "sigma_norm");
          data_out.add_data_vector(J, "J");

          Vector<double> soln(solution_n_volume.size());
          for (unsigned int i = 0; i < soln.size(); ++i)
            soln(i) = solution_n_volume(i);

          MappingQEulerian<spacedim> q_mapping(poly_order, soln,
              volume_dof_handler);
          data_out.build_patches(q_mapping, poly_order);

          std::ostringstream filename;
          filename << "data_out/solution-" << the_time.get_timestep() << ".vtu";

          std::ofstream output(filename.str().c_str());
          data_out.write_vtu(output);

          if (parameters.postprocess_qp_data == true)
            {
              // write the point data to file
              std::ostringstream filename_quad_point_data;
              filename_quad_point_data << "data_out/qp_data-"
                  << the_time.get_timestep() << ".vtk";
              const unsigned int n_quad_points_in_triangulation =
                  n_q_points_volume * volume_triangulation.n_active_cells();

              std::ofstream out(filename_quad_point_data.str().c_str(),
                  std::ofstream::out | std::ofstream::trunc);
              out << "# vtk DataFile Version 3.0" << std::endl;
              out << "# StressTensorInEveryIntegrationPoint" << std::endl;
              out << "ASCII" << std::endl;
              out << "DATASET UNSTRUCTURED_GRID" << std::endl;
              out << std::endl;

              out << "POINTS " << n_quad_points_in_triangulation << " double"
                  << std::endl;
              for (unsigned int j = 0; j < n_quad_points_in_triangulation; j++)
                out << points[j][0] << " " << points[j][1] << " "
                    << points[j][2] << std::endl;

              out << "POINT_DATA " << n_quad_points_in_triangulation
                  << std::endl;

              out << "TENSORS P double" << std::endl;

              for (unsigned int qp = 0; qp < n_quad_points_in_triangulation;
                  qp++)
                {
                  for (unsigned int i = 0; i < spacedim; i++)
                    for (unsigned int j = 0; j < spacedim; j++)
                      out << P_qp[qp][i][j] << " ";
                  out << std::endl;
                }

              out << "TENSORS sigma double" << std::endl;

              for (unsigned int qp = 0; qp < n_quad_points_in_triangulation;
                  qp++)
                {
                  for (unsigned int i = 0; i < spacedim; i++)
                    for (unsigned int j = 0; j < spacedim; j++)
                      out << sigma_qp[qp][i][j] << " ";
                  out << std::endl;
                }

              out << "TENSORS F double" << std::endl;

              for (unsigned int qp = 0; qp < n_quad_points_in_triangulation;
                  qp++)
                {
                  for (unsigned int i = 0; i < spacedim; i++)
                    for (unsigned int j = 0; j < spacedim; j++)
                      out << F_qp[qp][i][j] << " ";
                  out << std::endl;
                }

              out.close();
            }
        }

      // surface
        {

          // spatial mapping
          MappingQEulerian<dim, Vector<double>, spacedim> q_mapping(poly_order,
              solution_n_surface, surface_dof_handler);

          // spatial fe_values
          FEValues<dim, spacedim> fe_values(q_mapping, surface_fe, qf_surface,
              update_values | update_gradients | update_normal_vectors
                  | update_quadrature_points | update_JxW_values);

          // average value of the Piola-Kirchhoff stress in the volume
          Vector<double> P_norm(surface_triangulation.n_active_cells());

          std::vector<Point<spacedim> > points;
          std::vector<Point<spacedim> > normals; // the normal on the surface
          std::vector<Tensor<2, spacedim> > P_qp;
          // the energy norm on the surface
          double P_F_norm_surface = 0.;

            {
              unsigned int count = 0;

              for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
                  surface_triangulation.begin_active();
                  cell != surface_triangulation.end(); ++cell, ++count)
                {
                  fe_values.reinit(cell);
                  std::vector<ContinuumPoint<spacedim> > &cell_point_history_surf =
                      surface_cp_map[cell];

                  Tensor<1, spacedim> normal;
                  Tensor<2, spacedim> P_avg;

                  for (unsigned int qq = 0; qq < n_q_points_surface; ++qq)
                    {
                      normal =
                          (cell_point_history_surf[qq]).get_normal_spatial();
                      points.push_back(fe_values.quadrature_point(qq));
                      normals.push_back(normal);
                      P_qp.push_back((cell_point_history_surf[qq]).get_P());
                      P_avg += (cell_point_history_surf[qq]).get_P();

                      const double F_P_2_cp = std::pow(
                          double_contract(cell_point_history_surf[qq].get_P(),
                              cell_point_history_surf[qq].get_F()), 2);

                      P_F_norm_surface += F_P_2_cp * fe_values.JxW(qq);

                      psi_surface += cell_point_history_surf[qq].get_psi()
                          * fe_values.JxW(qq);
                    }

                  P_avg *= (1.0 / n_q_points_surface);
                  P_norm(count) = P_avg.norm();

                }

            }

          P_F_norm_surface = std::sqrt(P_F_norm_surface);
//          std::cout << "\tP:F norm on surface:\t" << P_F_norm_surface
//              << std::endl;
//          std::cout << "\tEnergy on surface:\t" << std::fixed << std::setprecision(5) << std::scientific << psi_surface
//                 << std::endl;

          DataOut<dim, DoFHandler<dim, spacedim> > data_out;

          data_out.attach_dof_handler(surface_dof_handler);

          std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
              spacedim,
              DataComponentInterpretation::component_is_part_of_vector);
          std::vector<std::string> displacement(spacedim, "displacement");

          data_out.add_data_vector(solution_n_surface, displacement,
              DataOut<dim, DoFHandler<dim, spacedim> >::type_dof_data,
              data_component_interpretation);

          data_out.add_data_vector(P_norm, "P_norm",
              DataOut<dim, DoFHandler<dim, spacedim> >::type_cell_data);

          data_out.build_patches(q_mapping, poly_order);

          std::ostringstream filename;
          filename << "data_out/solution_surface-" << the_time.get_timestep()
              << ".vtu";

          std::ofstream output(filename.str().c_str());
          data_out.write_vtu(output);

          if (parameters.postprocess_qp_data == true)
            {

              // write the point data to file
              std::ostringstream filename_quad_point_data;
              filename_quad_point_data << "data_out/qp_data_surface-"
                  << the_time.get_timestep() << ".vtk";

              const unsigned int n_quad_points_in_triangulation =
                  n_q_points_surface * surface_triangulation.n_active_cells();

              std::ofstream out(filename_quad_point_data.str().c_str(),
                  std::ofstream::out | std::ofstream::trunc);
              out << "# vtk DataFile Version 3.0" << std::endl;
              out << "# StressTensorInEveryIntegrationPoint" << std::endl;
              out << "ASCII" << std::endl;
              out << "DATASET UNSTRUCTURED_GRID" << std::endl;
              out << std::endl;

              out << "POINTS " << n_quad_points_in_triangulation << " double"
                  << std::endl;
              for (unsigned int j = 0; j < n_quad_points_in_triangulation; j++)
                out << points[j][0] << " " << points[j][1] << " "
                    << points[j][2] << std::endl;

              out << "POINT_DATA " << n_quad_points_in_triangulation
                  << std::endl;

              out << "VECTORS n double" << std::endl;

              for (unsigned int qp = 0; qp < n_quad_points_in_triangulation;
                  qp++)
                {
                  for (unsigned int i = 0; i < spacedim; i++)
                    out << normals[qp][i] << " ";
                  out << std::endl;
                }

              out << "TENSORS P double" << std::endl;

              for (unsigned int qp = 0; qp < n_quad_points_in_triangulation;
                  qp++)
                {
                  for (unsigned int i = 0; i < spacedim; i++)
                    for (unsigned int j = 0; j < spacedim; j++)
                      out << P_qp[qp][i][j] << " ";
                  out << std::endl;
                }

              out.close();
            }

        }
//        std::cout << "\tTotal energy:\t\t" << std::fixed << std::setprecision(5) << std::scientific << psi_volume + psi_surface
//                << std::endl;
      timer.leave_subsection();

    }

  template <int spacedim>
    void
    Solid<spacedim>::setup_surface_volume_dof_map ()
    {

      // the surface cells
      typename DoFHandler<dim, spacedim>::active_cell_iterator cell =
          surface_dof_handler.begin_active(), endc = surface_dof_handler.end();

      // the cell on the boundary of the volume
      typename DoFHandler<spacedim, spacedim>::active_face_iterator face;

      // the vectors of dofs for the cell and on the surface
      std::vector<types::global_dof_index> cell_dof_indices(
          dofs_per_surface_cell);
      std::vector<types::global_dof_index> face_dof_indices(
          dofs_per_surface_cell);

      for (; cell != endc; ++cell) // loop over all the cells on the surface
        {
          // get the face of the corresponding volume cells
          face = surface_to_volume_dof_iterator_map[cell];

          cell->get_dof_indices(cell_dof_indices);
          face->get_dof_indices(face_dof_indices);
          // map the degrees of freedom
          for (unsigned int ii = 0; ii < dofs_per_surface_cell; ii++)
            {
              surface_to_volume_dof_map[cell_dof_indices[ii]] =
                  face_dof_indices[ii];
            }
        }

      ExcDimensionMismatch(surface_to_volume_dof_iterator_map.size(),
          surface_dof_handler.n_dofs());
    }

////////////////////////////

}

int
main (
    int argc, char *argv[])
{
  using namespace dealii;
  using namespace Surface_Elasticity;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
      dealii::numbers::invalid_unsigned_int);

  try
    {
      deallog.depth_console(0);
      std::cout << "Finite surface elasticity" << std::endl;
      std::cout << "-------------------------" << std::endl;

      Solid<3> continuum("parameters/parameters.prm");
      continuum.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
          << "----------------------------------------------------"
          << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
          << std::endl << "Aborting!" << std::endl
          << "----------------------------------------------------"
          << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
          << "----------------------------------------------------"
          << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
          << "----------------------------------------------------"
          << std::endl;
      return 1;
    }

  return 0;
}
