using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActiveRagdoll : MonoBehaviour
{
    private JointController[] JointControllers;
    private Rigidbody PhysicalTorso;
    [SerializeField] private Animator physicalAnimator;

    private void OnValidate()
    {
        if(PhysicalTorso == null)
            PhysicalTorso = physicalAnimator.GetBoneTransform(HumanBodyBones.Hips).GetComponent<Rigidbody>();
    }

    // Start is called before the first frame update
    void Awake()
    {
        if (JointControllers == null) 
            JointControllers = PhysicalTorso.GetComponentsInChildren<JointController>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        for (int i = 0; i < JointControllers.Length; i++)
        {
            ConfigurableJointExtensions.SetTargetRotationLocal(JointControllers[i].joint, JointControllers[i].targetLocalRotation, JointControllers[i].initialLocalRotation);
        }
    }
}
