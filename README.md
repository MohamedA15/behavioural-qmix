PINN-Inspired Behavioural Constraints for
Cooperative Multi-Agent Reinforcement Learning

It integrates behavioural constraints into the QMIX temporal-difference loss to improve active participation, collision avoidance, and spatial generalisation in Level-Based Foraging (LBF).

Code Location & Key Files
GitHub Repository: MohamedA15/behavioural-qmix
Behavioural Constraints Implementation: src/constraints.py (Contains Active Participation, Interference Avoidance, Cooperative Commitment, and Goal Alignment losses)
Training Entry Point: src/main.py

 Repository Structure
 
```text
behavioural-qmix/
├── src/
│   ├── constraints.py   #  Core behavioural constraint loss implementations
│   ├── learners/        # QMIX & Behavioural QMIX loss routines
│   ├── controllers/     # Multi-agent controllers
│   ├── envs/            # Environment wrappers
│   └── main.py          # Main execution script
├── config/              # Experiment & environment YAML configs
└── requirements.txt     # Python dependencies

```

# Clone repo
git clone https://github.com/MohamedA15/behavioural-qmix.git
cd behavioural-qmix

# Install requirements
pip install -r requirements.txt

# Launch experiment
python src/main.py --config=behavioural_qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
