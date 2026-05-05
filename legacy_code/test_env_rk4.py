from src.env_rk import Environment
from src.env_rk_jax import JAXDDEEnv, JAXEnvWrapper
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t

class TestEnvironmentRK4(unittest.TestCase):
    def test_delay(self):
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A_1 = np.array([[0., 0.], [0., -0.5]])
        Q = np.eye(2)
        R = np.eye(1)
        delay = np.array([0.5, 0.5])
        step_size = 0.1
        x0 = np.array([1., 0.])
        
        env = Environment(A=A, B=B, Q=Q, R=R, delay=delay, step_size=step_size, x0=x0, A1=A_1)
        
        self.assertIsNotNone(env._buffer)
        self.assertEqual(env._buffer.size, int(np.ceil(np.max(delay) / (step_size / env.resolution)))+1)
        self.assertTrue(np.array_equal(env.x0, x0))
        pos = []
        time = []
        for i in range(100):
            t, x_t ,_ =env.step(np.array([0.]))
            pos.append(x_t)
            time.append(t)
        pos = np.array(pos)
        time = np.array(time)
        print(time.shape)
        print(pos.shape)
        plt.plot(time, pos[:,0], label='Position with delay')
        plt.plot(time, pos[:,1], label='Velocity with delay')
        A_1_zero = np.zeros_like(A_1)
        env_without_delay = Environment(A=A, B=B, Q=Q, R=R, delay=np.zeros_like(delay), step_size=step_size, x0 = x0, A1=A_1_zero)
        pos = []
        time = []
        for i in range(100):
            t, x_t ,_ =env_without_delay.step(np.array([0.]))
            pos.append(x_t)
            time.append(t)
        pos = np.array(pos)
        time = np.array(time)
        plt.plot(time, pos[:,0], label='Position without delay')
        plt.plot(time, pos[:,1], label='Velocity without delay')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('RK4 Integration with and without Delay')
        plt.legend()
        plt.show()
    
    def test_comparison_jitcdde_no_delay(self):
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A1 = np.zeros_like(A)
        delay = np.array([0., 0.])
        Q = np.eye(2)
        R = np.eye(1)
        step_size = 0.1
        x0 = np.array([1., 0.])
        
        env = Environment(A=A, B=B, Q=Q, R=R, step_size=step_size, x0=x0, A1=A1, delay=delay)
        
        pos_rk4 = []
        time_rk4 = []
        for i in range(100):
            t, x_t ,_ =env.step(np.array([0.]))
            pos_rk4.append(x_t)
            time_rk4.append(t)
        pos_rk4 = np.array(pos_rk4)
        time_rk4 = np.array(time_rk4)
        
        f = [y(1), -y(0) - y(1)]
        DDE = jitcdde(f)
        DDE.constant_past(x0)
        DDE.step_on_discontinuities()
        
        pos_jitcdde = []
        time_jitcdde = []
        for i in range(100):
            t = i * step_size
            x_t = DDE.integrate(t)
            pos_jitcdde.append(x_t)
            time_jitcdde.append(t)
        pos_jitcdde = np.array(pos_jitcdde)
        time_jitcdde = np.array(time_jitcdde)
        
        plt.plot(time_rk4, pos_rk4[:,0], label='RK4 Position')
        plt.plot(time_jitcdde, pos_jitcdde[:,0], '--', label='JITCDDE Position')
        plt.plot(time_rk4, pos_rk4[:,1], label='RK4 Velocity')
        plt.plot(time_jitcdde, pos_jitcdde[:,1], '--', label='JITCDDE Velocity')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('RK4 vs JITCDDE Integration')
        plt.legend()
        plt.show()
    
    def test_comparison_jitcdde_delay(self):
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A1 = np.array([[0.5, 0.], 
               [-0.5, 0.]])
        delay = np.array([0.5, 0.5])
        Q = np.eye(2)
        R = np.eye(1)
        step_size = 0.1
        x0 = np.array([1., 0.])
        
        env = Environment(A=A, B=B, Q=Q, R=R, step_size=step_size, x0=x0, A1=A1, delay=delay, resolution=20)
        
        pos_rk4 = []
        time_rk4 = []
        for i in range(100):
            time, x_t ,_ =env.step(np.array([0.]))
            pos_rk4.append(x_t)
            time_rk4.append(time)
        pos_rk4 = np.array(pos_rk4)
        time_rk4 = np.array(time_rk4)
        
        f = [y(1) + 0.5 * y(0, t - 0.5), -y(0) - y(1) - 0.5 * y(0, t - 0.5)]
        DDE = jitcdde(f)
        DDE.constant_past(x0)
        DDE.step_on_discontinuities()
        
        pos_jitcdde = []
        time_jitcdde = []
        for i in range(100):
            time = i * step_size
            x_t = DDE.integrate(time)
            pos_jitcdde.append(x_t)
            time_jitcdde.append(time)
        pos_jitcdde = np.array(pos_jitcdde)
        time_jitcdde = np.array(time_jitcdde)
        
        plt.plot(time_rk4, pos_rk4[:,0], label='RK4 Position')
        plt.plot(time_jitcdde, pos_jitcdde[:,0], '--', label='JITCDDE Position')
        plt.plot(time_rk4, pos_rk4[:,1], label='RK4 Velocity')
        plt.plot(time_jitcdde, pos_jitcdde[:,1], '--', label='JITCDDE Velocity')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('RK4 vs JITCDDE Integration')
        plt.legend()
        plt.show()


class TestJAXEnvironment(unittest.TestCase):
    """Tests for the JAX-based DDE environment."""
    
    def test_jax_env_reset(self):
        """Test that reset() properly initializes the JAX environment."""
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A1 = np.array([[0.5, 0.], [-0.5, 0.]])
        delay = np.array([0.5, 0.5])
        Q = np.eye(2)
        R = np.eye(1)
        step_size = 0.1
        resolution = 20
        x0 = np.array([1., 0.])
        
        env = JAXDDEEnv(A=A, B=B, A1=A1, delay=delay, Q=Q, R=R, 
                        step_size=step_size, resolution=resolution)
        wrapper = JAXEnvWrapper(env)
        
        # First reset
        key = jax.random.PRNGKey(0)
        state_x = wrapper.reset(key, x0=x0, t0=0.0)
        
        self.assertEqual(state_x.shape, (2,))
        np.testing.assert_array_almost_equal(state_x, x0)
        self.assertEqual(len(wrapper._data), 1)
        self.assertEqual(len(wrapper._time), 1)
        self.assertAlmostEqual(wrapper._time[0], 0.0)
        
        # Step a few times
        state = wrapper.state
        for _ in range(5):
            t, x_next, reward = wrapper.step(state, jnp.zeros((1,)))
            state = wrapper.state
        
        # Second reset - should clear history
        key2 = jax.random.PRNGKey(42)
        x0_new = np.array([0.5, -0.5])
        state_x2 = wrapper.reset(key2, x0=x0_new, t0=0.0)
        
        np.testing.assert_array_almost_equal(state_x2, x0_new)
        self.assertEqual(len(wrapper._data), 1)  # History cleared
        self.assertEqual(len(wrapper._time), 1)
        print("✓ JAX env reset test passed")
    
    def test_jax_comparison_jitcdde_delay(self):
        """Compare JAX DDE env with jitcdde for delayed system."""
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A1 = np.array([[0.5, 0.], [-0.5, 0.]])
        delay = np.array([0.5, 0.5])
        Q = np.eye(2)
        R = np.eye(1)
        step_size = 0.1
        resolution = 20
        x0 = np.array([1., 0.])
        
        # JAX environment
        env_jax = JAXDDEEnv(A=A, B=B, A1=A1, delay=delay, Q=Q, R=R,
                            step_size=step_size, resolution=resolution)
        wrapper = JAXEnvWrapper(env_jax)
        
        key = jax.random.PRNGKey(0)
        wrapper.reset(key, x0=x0, t0=0.0)
        
        pos_jax = []
        time_jax = []
        state = wrapper.state
        for i in range(100):
            t_step, x_t, _ = wrapper.step(state, jnp.zeros((1,)))
            state = wrapper.state
            pos_jax.append(np.array(x_t))
            time_jax.append(float(t_step))
        pos_jax = np.array(pos_jax)
        time_jax = np.array(time_jax)
        
        # jitcdde reference
        f = [y(1) + 0.5 * y(0, t - 0.5), -y(0) - y(1) - 0.5 * y(0, t - 0.5)]
        DDE = jitcdde(f)
        DDE.constant_past(x0)
        DDE.step_on_discontinuities()
        
        pos_jitcdde = []
        time_jitcdde = []
        for i in range(100):
            t_ref = i * step_size
            x_t = DDE.integrate(t_ref)
            pos_jitcdde.append(x_t)
            time_jitcdde.append(t_ref)
        pos_jitcdde = np.array(pos_jitcdde)
        time_jitcdde = np.array(time_jitcdde)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(time_jax, pos_jax[:, 0], label='JAX Position')
        plt.plot(time_jitcdde, pos_jitcdde[:, 0], '--', label='JITCDDE Position')
        plt.plot(time_jax, pos_jax[:, 1], label='JAX Velocity')
        plt.plot(time_jitcdde, pos_jitcdde[:, 1], '--', label='JITCDDE Velocity')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('JAX DDE vs JITCDDE Integration (with delay)')
        plt.legend()
        plt.grid(True)
        plt.show()

        
        
        # Check that results are reasonably close
        max_diff = np.max(np.abs(pos_jax - pos_jitcdde))
        print(f"Max difference JAX vs JITCDDE: {max_diff:.6f}")
        self.assertLess(max_diff, 0.2, "JAX and JITCDDE results differ too much")
        new_x0 = jnp.array([1, 1.])
        wrapper.reset(key, x0=new_x0, t0=0.0)
        pos_jax = []
        time_jax = []
        state = wrapper.state
        for i in range(100):
            t_step, x_t, _ = wrapper.step(state, jnp.zeros((1,)))
            state = wrapper.state
            pos_jax.append(np.array(x_t))
            time_jax.append(float(t_step))
        pos_jax = np.array(pos_jax)
        time_jax = np.array(time_jax)
        
        # jitcdde reference
        f = [y(1) + 0.5 * y(0, t - 0.5), -y(0) - y(1) - 0.5 * y(0, t - 0.5)]
        DDE = jitcdde(f)
        DDE.constant_past(new_x0)
        DDE.step_on_discontinuities()
        
        pos_jitcdde = []
        time_jitcdde = []
        for i in range(100):
            t_ref = i * step_size
            x_t = DDE.integrate(t_ref)
            pos_jitcdde.append(x_t)
            time_jitcdde.append(t_ref)
        pos_jitcdde = np.array(pos_jitcdde)
        time_jitcdde = np.array(time_jitcdde)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(time_jax, pos_jax[:, 0], label='JAX Position')
        plt.plot(time_jitcdde, pos_jitcdde[:, 0], '--', label='JITCDDE Position')
        plt.plot(time_jax, pos_jax[:, 1], label='JAX Velocity')
        plt.plot(time_jitcdde, pos_jitcdde[:, 1], '--', label='JITCDDE Velocity')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('JAX DDE vs JITCDDE Integration (with delay)')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def test_jax_vs_numpy_env(self):
        """Compare JAX env with NumPy env for same parameters."""
        A = np.array([[0., 1.], [-1., -1.]])
        B = np.array([[0.], [1.]])
        A1 = np.array([[0.5, 0.], [-0.5, 0.]])
        delay = np.array([0.5, 0.5])
        Q = np.eye(2)
        R = np.eye(1)
        step_size = 0.1
        resolution = 20
        x0 = np.array([1., 0.])
        
        # NumPy environment
        env_np = Environment(A=A, B=B, Q=Q, R=R, step_size=step_size, 
                             x0=x0, A1=A1, delay=delay, resolution=resolution)
        
        pos_np = []
        time_np = []
        for i in range(100):
            t_np, x_t, _ = env_np.step(np.array([0.]))
            pos_np.append(x_t)
            time_np.append(t_np)
        pos_np = np.array(pos_np)
        time_np = np.array(time_np)
        
        # JAX environment
        env_jax = JAXDDEEnv(A=A, B=B, A1=A1, delay=delay, Q=Q, R=R,
                            step_size=step_size, resolution=resolution)
        wrapper = JAXEnvWrapper(env_jax)
        
        key = jax.random.PRNGKey(0)
        wrapper.reset(key, x0=x0, t0=0.0)
        
        pos_jax = []
        time_jax = []
        state = wrapper.state
        for i in range(100):
            t_jax, x_t, _ = wrapper.step(state, jnp.zeros((1,)))
            state = wrapper.state
            pos_jax.append(np.array(x_t))
            time_jax.append(float(t_jax))
        pos_jax = np.array(pos_jax)
        time_jax = np.array(time_jax)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(time_np, pos_np[:, 0], label='NumPy Position')
        plt.plot(time_jax, pos_jax[:, 0], '--', label='JAX Position')
        plt.plot(time_np, pos_np[:, 1], label='NumPy Velocity')
        plt.plot(time_jax, pos_jax[:, 1], '--', label='JAX Velocity')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('NumPy env vs JAX env')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        max_diff = np.max(np.abs(pos_np - pos_jax))
        print(f"Max difference NumPy vs JAX: {max_diff:.6f}")
        self.assertLess(max_diff, 1e-5, "NumPy and JAX envs differ too much")

    def test_jax_env_reward(self):
        A = np.array([[-0.1]])
        B = np.array([[1.0 ]])
        A1 = np.zeros_like(A)
        delay = np.array([0.0])
        Q = np.eye(1)
        R = np.eye(1)
        step_size = 0.01
        resolution = 10
        x0 = np.array([3.0])
        env_jax = JAXDDEEnv(A=A, B=B, A1=A1, delay=delay, Q=Q, R=R,
                            step_size=step_size, resolution=resolution)
        wrapper = JAXEnvWrapper(env_jax)
        key = jax.random.PRNGKey(0)
        wrapper.reset(key, x0=x0, t0=0.0)
        state = wrapper.state
        total_reward = 0.0
        for _ in range(200):
            u = -0.9*state[0]
            t_jax, x_t, reward = wrapper.step(state, u)
            state = wrapper.state
            total_reward += reward * step_size
        expected_reward = 0.0
        x_curr = x0.copy()
        for _ in range(200):
            x_curr = x_curr + step_size * (-0.1 * x_curr) + step_size * (1.0 * (-0.9 * x_curr))
            u = -0.9 * x_curr
            expected_reward += -(x_curr.T @ Q @ x_curr + u.T @ R @ u) * step_size
        print(f"Total JAX reward: {total_reward}, Expected reward: {expected_reward}")


if __name__ == '__main__':
    unittest.main()
