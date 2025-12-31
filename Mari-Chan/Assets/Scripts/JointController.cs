using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(ConfigurableJoint))]
public class JointController : MonoBehaviour
{
    public Quaternion targetLocalRotation = Quaternion.identity;

    [Header("PD Gains")]
    public float spring = 500f;
    public float damper = 50f;
    public float maxForce = 1000f;

    public ConfigurableJoint joint;
    public Quaternion initialLocalRotation;
    
    // Start is called before the first frame update
    void Awake()
    {
        joint = GetComponent<ConfigurableJoint>();
        initialLocalRotation = transform.localRotation;
        SetupJointDrive();
    }

    void SetupJointDrive()
    {
        JointDrive drive = new JointDrive
        {
            positionSpring = spring,
            positionDamper = damper,
            maximumForce = maxForce,
        };

        joint.rotationDriveMode = RotationDriveMode.Slerp;
        joint.slerpDrive = drive;
    }

/*    // Update is called once per frame
    void FixedUpdate()
    {
        Quaternion desired = targetLocalRotation * initialLocalRotation;
        joint.targetRotation = Quaternion.Inverse(desired);
    }*/
}
