import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from src.envs.platoon import PlatoonEnv
from src.envs.env_rk_jax import JAXEnvWrapper
from src.solvers.delayed_lqr import augmented_discrete_lqr
from src.representations.factory import make_representation, RepresentationBuffer

env=PlatoonEnv(n_vehicles=5,alpha=3.0,beta=1.0,delay=0.2,step_size=0.05,resolution=4)
n,dt=env.N,env.step_size; sigma=float(env.max_delay)
A,A1,B,Q,R=[np.array(getattr(env,k)) for k in 'A A1 B Q R'.split()]
dl=augmented_discrete_lqr(A,A1,B,Q,R,sigma,dt); k=dl.k_taps
tau=10.0; gamma=np.exp(-dt/tau); Rinv_BT=0.5*np.linalg.inv(R)@B.T
rng=np.random.default_rng(0)

def rollout(x0):
    w=JAXEnvWrapper(env,rng_key=42); w.reset(jax.random.PRNGKey(0),x0=jnp.array(x0),t0=0.0)
    win=[x0.copy() for _ in range(k+1)]; xs=[]; rs=[]; t=0.
    while t<15-1e-9:
        u=-dl.gain@np.concatenate(win[:k+1]); t,x,r=w.step(w.state,jnp.array(u)); x=np.array(x).reshape(-1)
        xs.append(x); rs.append(float(r)); win=[x]+win[:-1]
    return np.array(xs),np.array(rs)
ics=[np.zeros(n)]; ics[0][1]=0.5
for _ in range(11):
    x0=np.zeros(n); x0[1::2]=0.4*rng.standard_normal(n//2); ics.append(x0)
ROLL=[rollout(x0) for x0 in ics]

def make_windows(win_len):
    """Reconstruct agent windows (win_len taps, oldest..newest) along all rollouts + return-to-go."""
    W=[]; V=[]
    for xs,rs in ROLL:
        rtg=np.zeros(len(rs)); acc=0.
        for i in range(len(rs)-1,-1,-1): acc=rs[i]*dt+gamma*acc; rtg[i]=acc
        buf=[xs[0]]*win_len
        for j,x in enumerate(xs):
            buf=buf[1:]+[x]; W.append(np.array(buf)); V.append(rtg[j])
    return np.array(W), np.array(V)   # W: (Nsteps, win_len, n)

def fit_and_grad_fns(kind, win_len, **kw):
    rep=make_representation(kind,window_length=win_len,n_state=n,**kw)
    buf=RepresentationBuffer(rep,window_length=win_len,n_state=n)
    sig_fn=buf._jit_compute_sig
    W,V=make_windows(win_len)
    Phi=np.array([np.array(sig_fn(jnp.asarray(w))) for w in W])
    d=Phi.shape[1]; lam=1e-6*np.trace(Phi.T@Phi)/d
    theta=np.linalg.solve(Phi.T@Phi+lam*np.eye(d), Phi.T@V); th=jnp.asarray(theta)
    grad_fn=jax.jit(jax.grad(lambda p: jnp.dot(th, sig_fn(p))))
    return buf,grad_fn,W
def u_sig(grad_fn, window):  # value-gradient control from the fitted critic
    g=np.array(grad_fn(jnp.asarray(window)))[-1]; return Rinv_BT@g
def u_oracle(window):        # oracle control on the last k+1 taps (newest first)
    xi=np.concatenate(window[::-1][:k+1]); return -dl.gain@xi

print("Vertical-derivative CONTROL error vs off-manifold perturbation eta (cosine to oracle u*; 1=perfect direction).")
print("(state scale ~0.3; eta is added Gaussian noise std on the window)")
for win_len in [8, 16]:
  print(f"\n--- window_length={win_len} ---")
  print(f"{'representation':>16} | " + " ".join(f"eta={e:<5}" for e in [0.0,0.02,0.05,0.1,0.2]))
  for tag,kind,kw in [("markovian d2","markovian",dict(degree=2)),
                      ("signature d2","signature",dict(depth=2)),
                      ("signature d3","signature",dict(depth=3))]:
    buf,grad_fn,W=fit_and_grad_fns(kind,win_len,**kw)
    idx=rng.choice(len(W),60,replace=False)
    row=[]
    for eta in [0.0,0.02,0.05,0.1,0.2]:
        coss=[]
        for i in idx:
            wp=W[i]+eta*rng.standard_normal(W[i].shape)
            us=u_sig(grad_fn,wp); uo=u_oracle(wp)
            c=float(us@uo/(np.linalg.norm(us)*np.linalg.norm(uo)+1e-12)); coss.append(c)
        row.append(np.mean(coss))
    print(f"{tag:>16} | " + " ".join(f"{c:>+6.2f}   " for c in row))
