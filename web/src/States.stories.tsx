import type { Meta, StoryObj } from '@storybook/react';
import { Badge, Empty } from './App';
const meta:Meta={title:'Workspace/Evidence states'};
export default meta;
type Story=StoryObj;
export const EvidenceLevels:Story={render:()=> <div style={{display:'flex',gap:12}}><Badge tone="cyan">Observed</Badge><Badge tone="green">Recomputed</Badge><Badge tone="violet">Estimated</Badge><Badge tone="amber">Unavailable</Badge><Badge tone="danger">Contract failed</Badge></div>};
export const NoCapture:Story={render:()=> <Empty title="No committed runs" detail="Import a canonical trace or run ptm demo."/>};
export const EmptyFilter:Story={render:()=> <Empty title="No groups match this filter" detail="Clear the step or search filter to inspect more captured evidence."/>};
