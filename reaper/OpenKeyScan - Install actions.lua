-- OpenKeyScan: adds all the actions to the action list, so they do not have to be loaded
-- one by one. Load and run this script once; running it again does no harm.
local dir = ({ reaper.get_action_context() })[2]:match("^(.*[\\/])")

local ACTIONS = {
  "OpenKeyScan - Key and BPM of selected items.lua",
  "OpenKeyScan - Key and BPM, set project tempo.lua",
  "OpenKeyScan - Cycle BPM range.lua",
}

local added = 0
for i, name in ipairs(ACTIONS) do
  -- 0 is the Main section. Only the last call saves the list, so it is written to disk once.
  if reaper.AddRemoveReaScript(true, 0, dir .. name, i == #ACTIONS) ~= 0 then added = added + 1 end
end

local msg = added == #ACTIONS and "OpenKeyScan actions added to the action list: " .. added .. "."
  or "Added " .. added .. " of " .. #ACTIONS .. " OpenKeyScan actions. Unpack the whole archive and run this script again."
if reaper.osara_outputMessage then
  reaper.osara_outputMessage(msg)
else
  reaper.MB(msg, "OpenKeyScan", 0)
end
