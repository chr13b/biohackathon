Notable: my configs/reference.json in the repo originally had batch_size: 32 (matching the default I saw in the Apheris UI's settings
JSON), but the actual reference run that completed (finetune_7taSY6i8b, "test-with-mg-zn") used batch_size: 1. I matched that for our run
to stay consistent with the working reference. With crop_size: 384 and a batch of 1, gradient updates are made per cropped structure
sample.

### Last output:
/btw how is the fine-tuning going? What are you currently doing?

- API client logic is proven on the test jobs (CIF auto-parse, MSA upload, start endpoint).
- A free baseline from the published reference run (finetune_7taSY6i8b): PL LDDT 0.6555, IP LDDT 0.9429 on the 17-val superset that
includes our 8 held-out.

So: FT is not progressing yet; the bisection retry is the next concrete step but not in flight.

#### Idea: 
- we can maybe add 9 more 5S* augmentations


### Btw outputs:
/btw do you also save the fine-tuned model afterwards?

    edit additional_weights.json, run ./deploy_apherisfold && ./deploy_apherisfold diagnose.

So after the FT run lands, three things are saved: (a) all per-step checkpoints stored by Apheris under the job's output path on the VM,
(b) a deployable weight version registered for inference once deploy is called, and (c) the option to download the checkpoint file
directly and deploy it via the script.


/btw where does the train and val data live from the current run?

- Committed as 22b0804 on main, visible at github.com/chr13b/biohackathon.

Note the mismatch between the repo (44 train) and what was actually uploaded for the running FT job (35 train): the repo holds the full
rebalanced split as originally planned, while the live FT consumed only the 5S*-only subset after we dropped the 9 non-5S* augmentations
to get past the MonitorDatasetGeneration failure.


#### Submission Prompt
I will probably still have to wait for the whole pipeline to finish, can you in the meanwhile make the whole pipeline somewhat 
reproducible since it is working very well (since the crash we were able to recover the checkpoint which can be found in checkpooint_of_first_run directory). This includes making the really used dataset available (means that we only keep the data that   
is really used for the training process) including also the config and setup. In that matter it would be great if someone from the outside
could use the repo to reproduce our results easily. I want everything in order to be able to run this pipeline to be inside of this git     
repo. Therefore, it is also required to have a proper and an easy to understand README.md file which explains our approach and how to use   
it using the directory structure.  
Furthermore, I want you to give a step by step explanation of what has been done such that this could also be presented in a
presentation, to an audience that needs clear and easy to understand language. Favorably, this can be supported by good visuals.
