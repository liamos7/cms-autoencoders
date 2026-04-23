import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_PATH = "outputs/nae_phase2_zb_dim20/events.out.tfevents.1776911756.adroit-h11g3.log"

ea = EventAccumulator(LOG_PATH)
ea.Reload()


def get_steps_values(tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("NAE Phase 2 Training (ZB, dim=20)", fontsize=14)

ax = axes[0, 0]
steps, values = get_steps_values("energy/pos_energy_")
ax.plot(steps, values)
steps, values = get_steps_values("energy/neg_energy_")
ax.plot(steps, values)
ax.set_title("Energies")
ax.set_xlabel("Step")
ax.set_ylabel("Energy")
ax.legend(["Positive", "Negative"])

ax = axes[0, 1]
steps, values = get_steps_values("roc_auc_")
ax.plot(steps, values)
ax.set_title("ROC AUC")
ax.set_xlabel("Step")
ax.set_ylabel("AUC")

ax = axes[1, 0]
steps, values = get_steps_values("loss/val_loss_")
ax.plot(steps, values)
ax.set_title("Validation Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")

ax = axes[1, 1]
steps, values = get_steps_values("loss/train_loss_")
ax.plot(steps, values)
ax.set_title("Train Loss")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")

plt.tight_layout()
plt.savefig("outputs/nae_phase2_zb_dim20/training_curves.png", dpi=150)
print("Saved to outputs/nae_phase2_zb_dim20/training_curves.png")
