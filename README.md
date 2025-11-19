### 1. Clone the repository:
   ```bash
   git clone https://github.com/TmKim0812/Intro_Robotics_Final_Prj.git
   cd Intro_Robotics_Final_Prj
   ```

### 2. Create a virtual environment(Optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

### 3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 4. Control the robot with Jacobian-based velocity control

   ```bash
   python3 Franka_with_Obstacle.py
   ```
   -> 3 obstacles(box-shaped), one goal point(green sphere) / The end effector can reach to the goal point, but not implemented avoiding obstacles yet