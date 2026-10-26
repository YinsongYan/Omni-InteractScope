"""Only user-visible needs enter prompts; annotations are never model input."""
# Retain the original prompt marker so this package rename does not change model inputs.
PREFIX = '[CAT10_ASSISTANCE_POLICY_V1]\n'


def request_text(it, native=False):
    if it.target_group not in ('visually_impaired', 'hearing_impaired'):
        raise ValueError('missing/unknown target_group')
    lang = it.qa_language
    if lang not in ('en', 'zh'):
        raise ValueError('missing/unknown qa_language')
    profile = ('The user cannot see the scene. Convey relevant visual information in words.'
               if it.target_group == 'visually_impaired' else
               'The user cannot hear the scene. Convey relevant audible events and speech in text; '
               'use the original audio and supporting visible clues.')
    parts = [PREFIX.rstrip(), profile, 'Reply in ' + ('English.' if lang == 'en' else 'Chinese.'),
             'Use only the provided scene evidence up to now. Be accurate, concise and actionable. '
             'Do not guess unseen objects, text, directions, distances or unheard sounds.']
    if it.sub_task_type == 'A':
        parts.append('Monitor for useful assistance and timely safety reminders. State what is happening '
                     'and an appropriate action. Do not wait for a question. Avoid redundant reminders.')
    elif it.sub_task_type == 'B':
        parts.append('Answer the user’s question once the evidence is sufficient.')
    elif it.sub_task_type == 'C':
        parts.append('Guide the user’s intention with short steps as the observed state changes. '
                     'Correct visible mistakes. Give only the step currently justified, not future steps.')
    else:
        raise ValueError('unknown T8 subtype')
    if native:
        parts.append('Stay silent when there is no useful new assistance to provide.')
    else:
        parts.append('If no useful new response is warranted now, output exactly ABSTAIN.')
    if it.sub_task_type != 'A':
        parts.append('User request: ' + it.query_text)
    return '\n'.join(parts)
